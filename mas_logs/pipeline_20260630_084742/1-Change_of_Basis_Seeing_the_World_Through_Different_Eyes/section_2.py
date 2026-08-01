from manim import *

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid (4x4 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 0.5 + j * 1
                y = 2.2 - i * 1
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def place_in_area(self, mobject, top_left, bottom_right, scale_factor=1.0):
        tl_pos = self.grid[top_left]
        br_pos = self.grid[bottom_right]
        
        # Calculate center of the area
        center_x = (tl_pos[0] + br_pos[0]) / 2
        center_y = (tl_pos[1] + br_pos[1]) / 2
        center = np.array([center_x, center_y, 0])
        
        mobject.scale(scale_factor)
        mobject.move_to(center)
        return mobject

class Section2Scene(TeachingScene):
    def construct(self):
        # Setup Title and Lecture Lines
        title_txt = "The Perspective Shift: Enter the Robot"
        lecture_lines = [
            'Meet Z-4. His world is tilted 45 degrees.', 
            'He sees the same point using different basis vectors.', 
            'His coordinates describe the same location in his language.'
        ]
        self.setup_layout(title_txt, lecture_lines)

        # Color definitions
        Z4_COLOR = "#00FF00"   # Green
        B1_COLOR = "#00FF00"   # Green
        B2_COLOR = "#FFFF00"   # Yellow
        STAR_COLOR = "#FFD700" # Gold
        BOB_COLOR = WHITE

        # === Animation for Lecture Line 1 ===
        # Highlight line 1 color
        self.play(self.lecture[0].animate.set_color(Z4_COLOR))

        # Asset: Z-4 robot [Asset: /mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/robot.svg]
        # Load, color, and rotate 45 degrees
        z4_robot = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/robot.svg")
        z4_robot.set_color(Z4_COLOR)
        z4_robot.rotate(45 * DEGREES)
        self.place_at_grid(z4_robot, "B2", scale_factor=0.6)

        # Bob's coordinate system (standard grid) centered at D3
        bob_plane = NumberPlane(
            x_range=[-2, 4, 1],
            y_range=[-2, 4, 1],
            background_line_style={"stroke_color": GREY_E, "stroke_opacity": 0.4}
        )
        self.place_at_grid(bob_plane, "D3")

        # Z-4's Tilted Basis Vectors
        origin_coords = self.grid["D3"]
        b1_target = self.grid["C4"]
        b2_target = self.grid["C2"]

        b1_vec = Arrow(origin_coords, b1_target, buff=0, color=B1_COLOR)
        b2_vec = Arrow(origin_coords, b2_target, buff=0, color=B2_COLOR)
        
        # Labels for basis vectors with layout fixes (Issue 32, 33)
        b1_label = Text("b1", color=B1_COLOR, font_size=22)
        self.place_at_grid(b1_label, "D5", scale_factor=0.5)
        
        b2_label = Text("b2", color=B2_COLOR, font_size=22)
        self.place_at_grid(b2_label, "B1", scale_factor=0.5)

        self.play(FadeIn(bob_plane))
        self.play(FadeIn(z4_robot, shift=DOWN))
        self.play(
            Create(b1_vec), 
            Create(b2_vec), 
            Write(b1_label), 
            Write(b2_label),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight line 2 color
        self.play(self.lecture[1].animate.set_color(B2_COLOR))

        # Gold star at Bob's (1,1) which corresponds to grid position C4
        star = Star(n=5, color=STAR_COLOR, fill_opacity=1)
        self.place_at_grid(star, "C4", scale_factor=0.2)
        
        # Coordinate label for Bob with layout fix (Issue 31)
        bob_coords = Text("(1, 1) Bob", color=BOB_COLOR, font_size=22)
        self.place_at_grid(bob_coords, "D4", scale_factor=0.6)

        self.play(FadeIn(star, scale=0.5))
        self.play(Write(bob_coords))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight line 3 color
        self.play(self.lecture[2].animate.set_color(STAR_COLOR))

        # Coordinate label for Z-4 with layout fix (Issue 31)
        z4_coords = Text("(1, 0) Z-4", color=Z4_COLOR, font_size=22)
        self.place_at_grid(z4_coords, "B4", scale_factor=0.6)

        # Visual indicator: highlighting the basis vector b1 as the axis of measurement
        focus_circle = Circle(radius=0.4, color=Z4_COLOR).move_to(self.grid["C4"])

        self.play(Create(focus_circle))
        self.play(Write(z4_coords))
        self.play(Indicate(z4_coords, color=Z4_COLOR))
        self.play(FadeOut(focus_circle))
        
        self.wait(2)

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

class Section1Scene(TeachingScene):
    def construct(self):
        # Define lecture lines
        lecture_lines = [
            "A basis is a set of building blocks.",
            "Meet Bob. He uses unit vectors i and j.",
            "To reach (3, 2), he takes 5 total steps."
        ]
        
        self.setup_layout("Prerequisite Check: The Standard Home", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE)
        
        # Create a standard square grid (axes_grid)
        # Using ranges that keep (0,0) near E2 when centered in A2-F6
        axes_grid = NumberPlane(
            x_range=[-1, 5, 1],
            y_range=[-1, 4, 1],
            background_line_style={
                "stroke_color": "#333333",
                "stroke_width": 2,
                "stroke_opacity": 0.6
            },
            axis_config={"include_tip": False, "stroke_color": "#555555"}
        )
        # Fix: Anchor axes_grid to visual grid system (Issue 28, 45)
        self.place_in_area(axes_grid, 'A2', 'F6', scale_factor=0.9)
        
        self.play(Create(axes_grid), run_time=1.5)
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(BLUE)
        
        # Unit vector i (Blue: #0072B2)
        i_vec = Arrow(axes_grid.c2p(0, 0), axes_grid.c2p(1, 0), buff=0, color="#0072B2")
        i_label = Text("i", color="#0072B2", font_size=24, slant=ITALIC)
        # Fix: Position unit vector label i (Issue 29, 45)
        self.place_at_grid(i_label, 'E3', scale_factor=0.6)
        
        # Unit vector j (Red: #FC6255)
        j_vec = Arrow(axes_grid.c2p(0, 0), axes_grid.c2p(0, 1), buff=0, color="#FC6255")
        j_label = Text("j", color="#FC6255", font_size=24, slant=ITALIC)
        # Fix: Position unit vector label j (Issue 29, 45)
        self.place_at_grid(j_label, 'D2', scale_factor=0.6)
        
        # Load and integrate Bob asset (Issue 25, 45)
        bob = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/bob.svg")
        bob.set_height(0.4)
        bob.move_to(axes_grid.c2p(0,0))

        self.play(
            GrowArrow(i_vec),
            Write(i_label),
            GrowArrow(j_vec),
            Write(j_label),
            FadeIn(bob)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        
        # Target point (3, 2)
        target_point = axes_grid.c2p(3, 2)
        intermediate_point = axes_grid.c2p(3, 0)
        
        # Path trace - matching vector colors
        path_x = Line(axes_grid.c2p(0,0), intermediate_point, color="#0072B2", stroke_width=4)
        path_y = Line(intermediate_point, target_point, color="#FC6255", stroke_width=4)
        
        # Bob's movement: 3 units along i then 2 units along j
        self.play(
            bob.animate.move_to(intermediate_point),
            Create(path_x),
            run_time=2
        )
        
        self.play(
            bob.animate.move_to(target_point),
            Create(path_y),
            run_time=1.5
        )
        
        # Final label: Bob (3, 2) (Issue 30, 45)
        bob_final_label = Text("Bob (3, 2)", font_size=24, color=YELLOW)
        self.place_in_area(bob_final_label, 'B5', 'C6', scale_factor=0.7)
        
        self.play(
            Write(bob_final_label)
        )
        self.wait(2)

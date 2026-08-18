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
        self.setup_layout("Defining the 3D Cross Product", [
            "3D cross product creates a perpendicular vector.",
            "Magnitude equals the spanned parallelogram area.",
            "Direction follows the Right-Hand Rule."
        ])

        # Colors
        blue = "#4682B4"
        highlight = "#FFD700"
        
        # Load assets
        hand = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/hand.svg")

        # Setup 3D elements
        axes = ThreeDAxes(x_range=[-1, 1], y_range=[-1, 1], z_range=[-1, 1])
        u = Arrow3D(start=ORIGIN, end=[1, 0, 0], color=blue)
        v = Arrow3D(start=ORIGIN, end=[0, 1, 0], color=blue)
        n = Arrow3D(start=ORIGIN, end=[0, 0, 1], color=RED)
        
        # Use VGroup to group for placement
        diagram = VGroup(axes, u, v, n)
        # Ensure points are float64 for compatibility with Manim transformations
        for mob in diagram.family_members_with_points():
            mob.points = mob.points.astype(np.float64)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(highlight))
        self.place_in_area(diagram, 'C2', 'E5', scale_factor=0.9)
        self.place_at_grid(hand.copy(), 'D2', scale_factor=0.3)
        self.play(Create(axes), Create(u), Create(v), Create(n))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE))
        self.play(self.lecture[1].animate.set_color(highlight))
        
        para = Polygon([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], color=GREY, fill_opacity=0.3)
        para.rotate(PI/2, axis=RIGHT, about_point=ORIGIN)
        para.move_to(diagram.get_center())
        
        point_labels = Text("Area", font_size=20)
        self.place_at_grid(point_labels, 'F2', scale_factor=0.7)
        
        self.play(Create(para), FadeIn(point_labels))
        self.place_at_grid(hand.copy(), 'E5', scale_factor=0.3)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE))
        self.play(self.lecture[2].animate.set_color(highlight))
        
        title_element = Text("Right-Hand Rule", font_size=24)
        self.place_at_grid(title_element, 'A3', scale_factor=1.0)
        
        self.play(Indicate(n), Write(title_element))
        self.wait(2)

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
        lecture_lines = [
            "Linear transformations stretch the space around us.",
            "Determinants measure the factor of area change.",
            "A unit square transformed creates a parallelogram.",
            "If area doubles, the determinant is two.",
            "It quantifies how space scales during transformation."
        ]
        self.setup_layout("The Determinant: Measuring Geometric Scaling", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Use SVG asset as requested in Issue 16
        square = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/square.svg")
        square.set_color(WHITE)
        self.place_at_grid(square, 'C2', scale_factor=0.6)
        self.play(Create(square))
        self.lecture[0].set_color("#00FFFF")

        # === Animation for Lecture Line 2 ===
        determinant_text = Text("Determinant", font_size=24, color="#FF00FF")
        self.place_at_grid(determinant_text, 'E4', scale_factor=0.8)
        self.play(Write(determinant_text))
        self.lecture[1].set_color("#FF00FF")

        # === Animation for Lecture Line 3 ===
        # Transform the square asset
        parallelogram = Polygon(
            np.array([-0.75, -0.75, 0]), 
            np.array([0.75, -0.75, 0]), 
            np.array([1.25, 0.75, 0]), 
            np.array([-0.25, 0.75, 0]),
            color=WHITE, fill_opacity=0.3
        )
        self.play(Transform(square, parallelogram))
        self.lecture[2].set_color("#00FF00")

        # === Animation for Lecture Line 4 ===
        area_label = Text("Area = 2", font_size=24, color="#FFFF00")
        self.place_at_grid(area_label, 'C3', scale_factor=0.7)
        self.play(Write(area_label))
        self.lecture[3].set_color("#FFFF00")

        # === Animation for Lecture Line 5 ===
        self.play(Indicate(determinant_text), Indicate(area_label))
        self.lecture[4].set_color("#FF8000")
        self.wait(1)

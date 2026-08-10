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
            "The determinant is the signed area of a parallelogram.",
            "Vectors form the sides of this geometric shape.",
            "Area scaling helps us understand linear transformations."
        ]
        self.setup_layout("Prerequisite Review: Determinants as Area/Volume", lecture_lines)
        
        # Basis Vectors
        v1 = Vector([2, 0], color=BLUE)
        v2 = Vector([0, 3], color=GREEN)
        vectors = VGroup(v1, v2)
        
        # === Animation for Lecture Line 1 ===
        # Fade in vectors v1=(2,0) and v2=(0,3)
        self.place_in_area(vectors, 'B4', 'E6', scale_factor=0.6)
        self.play(Create(vectors))
        self.lecture[0].set_color("#FFD700")

        # === Animation for Lecture Line 2 ===
        # Draw parallelogram spanned by v1 and v2, filled with color #444444
        parallelogram = Polygon(
            [0,0,0], [2,0,0], [2,3,0], [0,3,0],
            color="#FFFFFF", fill_opacity=0.3, fill_color="#444444"
        )
        self.place_in_area(parallelogram, 'B4', 'E6', scale_factor=0.6)
        self.play(DrawBorderThenFill(parallelogram))
        self.lecture[1].set_color("#00FFFF")

        # === Animation for Lecture Line 3 ===
        # Display text 'Area = Det(A) = 6' in color #FFFFFF
        formula = MathTex(r"\\text{Area} = \\det(A) = 6", color="#FFFFFF")
        self.place_at_grid(formula, 'F4', scale_factor=0.8)
        self.play(Write(formula))
        self.lecture[2].set_color("#FF4500")
        
        self.wait(1)

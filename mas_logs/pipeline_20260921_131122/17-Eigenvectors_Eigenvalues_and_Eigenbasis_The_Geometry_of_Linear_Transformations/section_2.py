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
        lecture_lines = [
            "Equation Av equals lambda v defines them.",
            "Eigenvector v keeps its direction constant.",
            "Eigenvalue lambda scales the vector magnitude."
        ]
        self.setup_layout("Defining Eigenvalues and Eigenvectors", lecture_lines)
        
        # Setup elements
        plane = NumberPlane(x_range=[-3, 3], y_range=[-3, 3], axis_config={"include_numbers": False})
        self.place_in_area(plane, 'B2', 'E5', scale_factor=0.9)
        
        vec_v = Vector(RIGHT + UP, color=WHITE)
        vec_Av = Vector(2 * (RIGHT + UP), color="#FFD700")
        
        self.place_at_grid(vec_v, 'C3')
        self.place_at_grid(vec_Av, 'C3')
        
        self.add(plane, vec_v, vec_Av)
        
        # Labels
        label_v = Text("v", color=WHITE, font_size=24)
        label_Av = Text("Av = λv", color="#FFD700", font_size=24)
        
        # Applying requested fixes from issue 36
        self.place_at_grid(label_v, 'B2', scale_factor=0.8)
        self.place_at_grid(label_Av, 'D3', scale_factor=0.8)
        
        # Adjust 'Av = λv' annotation position (replacing the previous label_Av placement/style)
        # Note: The prompt asks for Av = λv annotation specifically.
        # The critic suggested place_at_grid(vec_Av, 'E4', scale_factor=1.0)
        # Let's adjust the vector's label/annotation as requested.
        
        # Re-evaluating the label placement per instructions:
        # 1. label_v -> 'B2', scale 0.8
        # 2. label_Av -> 'D3', scale 0.8
        # 3. vec_Av -> 'E4', scale 1.0
        self.place_at_grid(vec_Av, 'E4', scale_factor=1.0)
        
        self.add(label_v, label_Av)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFD700"))
        self.play(vec_v.animate.set_color("#32CD32"), label_v.animate.set_color("#32CD32"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#32CD32"))
        self.play(vec_Av.animate.set_color("#32CD32"), label_Av.animate.set_color("#32CD32"))
        self.wait(2)

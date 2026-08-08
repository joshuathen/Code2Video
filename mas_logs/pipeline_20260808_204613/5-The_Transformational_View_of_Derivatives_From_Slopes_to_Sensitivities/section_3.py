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

class Section3Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Core Concept: The Derivative as a Local Linear Transformer", 
                          ["Derivatives are local linear transformers.", 
                           "Magnify a curve to see a line.", 
                           "The derivative is the local magnification factor.", 
                           "It translates input variations to output changes.", 
                           "Linear approximation defines the derivative."])
        
        # === Animation for Lecture Line 1 ===
        # Display linear transformation matrix acting on a vector
        matrix = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/matrix.svg")
        vector = Matrix([[1], [0]], left_bracket="[", right_bracket="]")
        self.place_at_grid(matrix, 'B2', scale_factor=0.6)
        self.place_at_grid(vector, 'B4', scale_factor=0.6)
        self.play(self.lecture[0].animate.set_color("#FFFFFF"), FadeIn(matrix), Write(vector))
        
        # === Animation for Lecture Line 2 ===
        # Animate vector rotating and scaling
        vector_new = Matrix([[1], [0.5]], left_bracket="[", right_bracket="]")
        self.place_at_grid(vector_new, 'B4', scale_factor=0.6)
        self.play(self.lecture[1].animate.set_color("#FF5733"), Transform(vector, vector_new))
        
        # === Animation for Lecture Line 3 ===
        # Show 'Derivative' as the local Jacobian matrix
        jacobian_label = Text("Derivative = J", font_size=24, color="#33FF57")
        self.place_at_grid(jacobian_label, 'C3', scale_factor=0.8)
        self.play(self.lecture[2].animate.set_color("#33FF57"), FadeIn(jacobian_label))
        
        # === Animation for Lecture Line 4 ===
        # Overlay local grid distortion to show linearity
        grid_dist = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        self.place_in_area(grid_dist, 'D4', 'E6', scale_factor=0.4)
        self.play(self.lecture[3].animate.set_color("#F3FF33"), FadeIn(grid_dist))
        
        # === Animation for Lecture Line 5 ===
        # Fade in label 'Local Linear Map'
        label = Text("Local Linear Map", font_size=28, color="#FFFFFF")
        self.place_at_grid(label, 'F4', scale_factor=0.7)
        self.play(self.lecture[4].animate.set_color("#FFFFFF"), FadeIn(label))
        
        self.wait(2)

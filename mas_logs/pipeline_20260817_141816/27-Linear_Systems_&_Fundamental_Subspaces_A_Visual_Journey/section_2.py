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
        self.setup_layout("The Inverse Matrix: The 'Undo' Button", [
            "Inverse matrices act as undo buttons.",
            "Matrix A transforms a vector's position.",
            "Inverse A brings the vector home."
        ])
        
        # Elements
        vector_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/vector.svg", color=WHITE)
        undo_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/undo.svg", color="#00FFFF")
        origin_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/origin.svg", color=GREEN)
        target_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/target.svg", color=RED)
        
        matrix_a = MathTex("A", color=WHITE)
        transformed_matrix = MathTex("A_{trans}", color="#AAAAAA")
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.place_at_grid(matrix_a, "B3", scale_factor=1.0)
        self.place_at_grid(transformed_matrix, "D3", scale_factor=1.0)
        self.place_at_grid(vector_icon, "B2", scale_factor=0.5)
        self.play(FadeIn(matrix_a), FadeIn(transformed_matrix), FadeIn(vector_icon))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(BLUE)
        
        point = origin_icon.copy()
        label = Text("Vector v", font_size=18, color=BLUE)
        self.place_at_grid(point, "B5")
        self.place_at_grid(label, "B5") # Fixed position (Constraint 25)
        self.place_at_grid(undo_icon, "D5", scale_factor=0.5)
        
        self.play(FadeIn(point), FadeIn(label), FadeIn(undo_icon))
        
        # Transform animation: point moves to target
        new_pos = self.grid["B6"]
        self.play(point.animate.move_to(new_pos), label.animate.move_to(new_pos + UP*0.4))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(RED)
        
        # Reverse animation: point returns to origin
        self.play(point.animate.move_to(self.grid["B5"]), label.animate.move_to(self.grid["B5"] + UP*0.4))
        self.wait(1)

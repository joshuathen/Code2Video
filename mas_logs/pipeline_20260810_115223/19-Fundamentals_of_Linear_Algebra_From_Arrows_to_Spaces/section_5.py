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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Conclusion: The Power of Linear Transformations", 
                          ["Transformations move, rotate, and resize spaces.", 
                           "These operations form the basis of AI.", 
                           "Like pinching to zoom on your screen."])
        
        # Assets
        smartphone = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/smartphone.svg")
        
        # Create grid of dots
        dots = VGroup(*[Dot(radius=0.05) for _ in range(25)])
        dots.arrange_in_grid(5, 5, buff=0.4)
        # Apply fix 34 & 40
        self.place_in_area(dots, 'B3', 'D5', scale_factor=0.5)
        
        # Add smartphone at the origin (center of the grid)
        smartphone.move_to(dots.get_center())
        self.add(smartphone)
        
        label = Text("Transformation", font_size=24)
        # Apply fix 35 & 40
        self.place_at_grid(label, 'B3', scale_factor=0.7)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00BFFF"))
        self.play(
            dots.animate.shift(RIGHT * 0.5).rotate(PI/6),
            smartphone.animate.shift(RIGHT * 0.5).rotate(PI/6),
            FadeIn(label)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#32CD32"))
        matrix = MathTex(r"\\begin{pmatrix} a & b \\ c & d \\end{pmatrix}", font_size=36)
        # Apply fix 33 & 40
        self.place_in_area(matrix, 'D4', 'E6', scale_factor=0.6)
        self.play(Write(matrix))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFD700"))
        self.play(
            dots.animate.scale(1.5),
            smartphone.animate.scale(1.5),
            run_time=1.5
        )
        self.wait(2)

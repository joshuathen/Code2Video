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
        self.setup_layout("Measuring Success: The Loss Function", [
            "Loss measures our prediction versus reality.",
            "Visualize error as a valley to descend.",
            "The goal is reaching the valley's bottom."
        ])
        
        # Define elements
        target_label = Text("T", color=GREEN)
        pred_label = Text("P", color=RED)
        self.place_at_grid(target_label, 'B3', scale_factor=0.6)
        self.place_at_grid(pred_label, 'B4', scale_factor=0.6)
        
        error_bar = Line(target_label.get_right(), pred_label.get_left(), color=WHITE)
        error_label = Text("Error", color=WHITE)
        self.place_at_grid(error_label, 'B2', scale_factor=0.7)
        
        # Background valley
        valley_img = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/valley.svg")
        self.place_in_area(valley_img, 'C1', 'F6', scale_factor=1.0)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#87CEFA"))
        self.play(FadeIn(valley_img))
        self.play(Create(target_label), Create(pred_label))
        self.play(Create(error_bar), Write(error_label))
        self.play(error_bar.animate.set_color("#FF4500"), error_label.animate.set_color("#FF4500"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#87CEFA"))
        # Draw a valley representation (parabola)
        valley_curve = FunctionGraph(lambda x: 0.5 * (x**2) - 1.5, x_range=[-2, 2], color=BLUE)
        self.place_in_area(valley_curve, 'C1', 'E6', scale_factor=0.9)
        self.play(Create(valley_curve))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#87CEFA"))
        # Shrink the error bar
        self.play(error_bar.animate.scale(0.2), error_label.animate.scale(0.5))
        # Move a ball to the bottom
        ball = Dot(color=YELLOW)
        ball.move_to(valley_curve.point_from_proportion(0.2))
        self.play(Create(ball))
        self.play(ball.animate.move_to(valley_curve.point_from_proportion(0.5)))
        self.wait(2)

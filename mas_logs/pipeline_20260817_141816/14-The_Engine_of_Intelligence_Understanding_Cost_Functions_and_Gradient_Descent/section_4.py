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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Hyperparameters: Step Size and Learning Rate", [
            "Learning rate determines the step size.", 
            "Large steps might overshoot the valley.", 
            "Tiny steps make learning agonizingly slow."
        ])
        
        # Load asset - SVG for valley
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/valley.svg]
        valley = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/valley.svg")
        # Address issue 30
        self.place_in_area(valley, 'A4', 'F6', scale_factor=0.6)
        self.add(valley)
        
        # Address issue 31
        point = Dot(color=YELLOW)
        self.place_at_grid(point, 'C5', scale_factor=1.2)
        self.add(point)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FF00"))
        # Visualize a step size vector in #00FF00
        step_vec = Arrow(start=point.get_center(), end=point.get_center() + RIGHT*0.5, color="#00FF00")
        self.play(Create(step_vec))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF4500"))
        # Demonstrate large steps overshooting
        self.play(point.animate.shift(RIGHT*1.5), run_time=1)
        self.play(FadeOut(step_vec))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#87CEEB"))
        # Demonstrate small steps converging smoothly
        self.play(point.animate.move_to(self.grid['E5']), run_time=2)
        self.wait(1)

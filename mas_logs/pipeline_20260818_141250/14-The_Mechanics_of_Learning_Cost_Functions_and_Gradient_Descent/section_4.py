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
        self.setup_layout("Dynamics: Learning Rate and Convergence", [
            "Learning rate defines our step size.",
            "Large steps might overshoot the valley.",
            "Small steps may be too slow."
        ])
        
        # Valley graphic
        valley = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/valley.svg")
        self.place_in_area(valley, 'B3', 'E6', scale_factor=0.5)
        self.add(valley)

        # === Animation for Lecture Line 1 ===
        lr_label = Text("Learning Rate: Step Size", color=WHITE, font_size=24)
        self.place_at_grid(lr_label, 'B2', scale_factor=0.7)
        self.play(Write(lr_label))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        large_step = Arrow(start=ORIGIN, end=RIGHT*1.5+UP*1, color="#FF0000")
        large_step.move_to(valley.get_center() + UP*0.5)
        self.play(Create(large_step), run_time=1.5)
        self.lecture[1].set_color("#FF0000")

        # === Animation for Lecture Line 3 ===
        small_step = Arrow(start=ORIGIN, end=RIGHT*0.3+DOWN*0.1, color="#00CCFF")
        small_step.move_to(valley.get_center() + DOWN*0.5)
        self.play(Create(small_step), run_time=1.5)
        self.lecture[2].set_color("#00CCFF")
        
        # Optimal point
        opt_label = Text("Optimal Step", color="#00FF00", font_size=20)
        self.place_at_grid(opt_label, 'E3', scale_factor=0.7)
        self.play(Write(opt_label))

        self.wait(2)

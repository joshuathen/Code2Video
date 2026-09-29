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
        self.setup_layout("The Role of Learning Rate", [
            "Learning rate controls our step size.",
            "Large steps may overshoot the minimum.",
            "Too small, and progress becomes painfully slow."
        ])
        
        # Pre-load assets
        slider = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/slider.svg")
        warning = Text("!", font_size=40, color=RED)
        target = Dot(color=GREEN).scale(2)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        lr_slider = self.place_at_grid(slider.copy(), 'B4', scale_factor=0.6)
        lr_label = Text("Learning Rate: Large", font_size=20, color=WHITE)
        self.place_at_grid(lr_label, 'A5', scale_factor=0.6)
        self.play(FadeIn(lr_slider), Write(lr_label))
        self.wait(3)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF0000")
        hiker = Circle(radius=0.2, color=BLUE, fill_opacity=1)
        self.place_at_grid(hiker, 'C4', scale_factor=0.7)
        self.place_at_grid(warning, 'B5', scale_factor=0.5)
        self.play(FadeIn(hiker), FadeIn(warning))
        self.wait(3)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FF00")
        lr_slider_opt = self.place_at_grid(slider.copy(), 'D4', scale_factor=0.6)
        opt_label = Text("Learning Rate: Optimal", font_size=20, color=GREEN)
        self.place_at_grid(opt_label, 'E4', scale_factor=0.6)
        self.play(Transform(lr_slider, lr_slider_opt), FadeIn(opt_label), FadeOut(hiker), FadeOut(warning))
        self.wait(3)

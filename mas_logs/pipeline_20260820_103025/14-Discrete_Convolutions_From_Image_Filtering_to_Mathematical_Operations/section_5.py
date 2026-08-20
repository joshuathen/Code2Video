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
        self.setup_layout("Summary & Synthesis", [
            "Convolution is a fundamental operation for feature extraction.",
            "Remember the steps: Flip, Slide, Multiply, Sum.",
            "It is the core engine behind AI vision systems."
        ])
        
        # Assets
        camera_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/camera.svg", color=WHITE)
        
        # Convolution visuals
        input_rect = Rectangle(width=2, height=2, color="#FFFFFF")
        kernel_rect = Rectangle(width=0.8, height=0.8, color="#FFFFFF")
        
        input_label = Text("Input", font_size=20)
        kernel_label = Text("Kernel", font_size=20)
        
        convolution_group = VGroup(input_rect, input_label, kernel_rect, kernel_label, camera_icon)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        
        # Fix 38: Place group in B4-E6
        self.place_in_area(convolution_group, 'B4', 'E6', scale_factor=0.6)
        self.play(FadeIn(convolution_group))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFCC00")
        
        # Logic: Labels at grid coords + next_to (B011, B020)
        input_label.next_to(input_rect, UP, buff=0.1).scale(0.7)
        kernel_label.next_to(kernel_rect, UP, buff=0.1).scale(0.7)
        
        # Slide animation
        self.play(kernel_rect.animate.shift(RIGHT * 0.5), kernel_label.animate.shift(RIGHT * 0.5))
        self.play(kernel_rect.animate.shift(LEFT * 0.5), kernel_label.animate.shift(LEFT * 0.5))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFFFF")
        
        # Fade out
        self.play(FadeOut(convolution_group))
        self.wait(1)

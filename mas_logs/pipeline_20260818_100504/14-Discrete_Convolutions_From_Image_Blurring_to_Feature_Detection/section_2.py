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
        self.setup_layout("The Mathematical Mechanism", [
            "Convolution is element-wise multiplication and summation.",
            "We flip the kernel for proper filtering.",
            "The kernel slides across the entire image."
        ])

        # Assets
        photo_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/photograph.svg")
        camera_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/camera.svg")

        # Objects
        image = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]], element_to_mobject=Integer)
        kernel = Matrix([[0, 1], [1, 0]], element_to_mobject=Integer)
        image.set_color(WHITE)
        kernel.set_color(YELLOW)

        self.place_at_grid(image, 'B3', scale_factor=0.6)
        self.place_at_grid(photo_icon, 'A3', scale_factor=0.5)
        self.place_at_grid(kernel, 'E5', scale_factor=0.7)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.play(FadeIn(image), FadeIn(photo_icon), FadeIn(kernel))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(BLUE)
        self.play(kernel.animate.rotate(PI))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(GREEN)
        output_val = Integer(0).set_color(GREEN)
        self.place_at_grid(output_val, 'F5', scale_factor=1.0)
        self.place_at_grid(camera_icon, 'F6', scale_factor=0.5)
        
        # Simple animation sequence
        self.play(
            kernel.animate.shift(LEFT * 0.5),
            output_val.animate.set_value(5),
            FadeIn(output_val),
            FadeIn(camera_icon)
        )
        self.wait(1)

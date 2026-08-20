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
        lecture_lines = ["Kernels can extract specific features.", "Edge detection highlights sharp transitions.", "Smoothing filters reduce image noise."]
        self.setup_layout("Real-World Application: Feature Extraction", lecture_lines)
        
        # Setup visuals
        # Represent the Kernel (3x3)
        kernel = VGroup(*[Square(side_length=0.5, color=BLUE) for _ in range(9)]).arrange_in_grid(rows=3, cols=3, buff=0)
        self.place_in_area(kernel, 'C2', 'D3', scale_factor=0.9)
        kernel_label = Text("Sobel Kernel", font_size=20, color=BLUE).next_to(kernel, UP)
        
        # Represent the image area
        image_obj = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/object.svg")
        image_area = Rectangle(width=3, height=3, color=WHITE)
        self.place_in_area(image_area, 'C4', 'D5', scale_factor=0.85)
        image_obj.match_width(image_area).match_height(image_area).move_to(image_area.get_center())
        image_label = Text("Image Input", font_size=20, color=WHITE).next_to(image_area, UP)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE), Create(kernel), Write(kernel_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW), Create(image_area), FadeIn(image_obj), Write(image_label))
        
        # Slide filter across image
        self.play(kernel.animate.move_to(image_area.get_center()), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(GREEN))
        feature_map = Square(side_length=1, color=GREEN, fill_opacity=0.5)
        self.place_at_grid(feature_map, 'D6', scale_factor=0.8)
        feature_label = Text("Edges Detected", font_size=20, color=GREEN).next_to(feature_map, DOWN)
        self.play(FadeIn(feature_map), Write(feature_label))
        self.wait(2)

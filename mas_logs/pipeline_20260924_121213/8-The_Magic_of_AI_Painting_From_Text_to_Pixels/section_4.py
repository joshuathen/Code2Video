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
        lecture_lines = ["CLIP provides the prompt guidance direction.", 
                         "Diffusion provides iterative denoising steps.", 
                         "Step by step, the sharp image emerges."]
        self.setup_layout("The Synthesis: Putting it Together", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Combine noise and text conditioning inputs
        self.lecture[0].set_color("#FFFFFF")
        noise_box = Rectangle(width=1.0, height=1.0, color=GREY).set_fill(GREY, opacity=0.3)
        text_box = Rectangle(width=1.0, height=1.0, color=BLUE).set_fill(BLUE, opacity=0.3)
        combined = VGroup(noise_box, text_box).arrange(RIGHT, buff=0.1)
        self.place_at_grid(combined, 'B2', scale_factor=0.6)
        self.play(FadeIn(combined))

        # === Animation for Lecture Line 2 ===
        # Visualize Transformer processing block [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/monitor.svg]
        self.lecture[1].set_color("#00FFFF")
        transformer_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/monitor.svg")
        transformer_box = Rectangle(width=1.5, height=1.5, color="#00FFFF").set_fill("#00FFFF", opacity=0.2)
        transformer = VGroup(transformer_box, transformer_icon)
        self.place_at_grid(transformer, 'D2', scale_factor=0.6)
        self.play(Create(transformer))

        # === Animation for Lecture Line 3 ===
        # Show iterative image refinement progress [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/camera.svg]
        self.lecture[2].set_color("#FF00FF")
        image_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/camera.svg")
        image_frame = Square(side_length=1.5, color=WHITE)
        image_content = VGroup(image_frame, image_icon)
        
        self.place_at_grid(image_content, 'E5', scale_factor=0.6)
        self.add(image_content)
        
        # Iteration animation
        target_icon = Dot(color=GREEN, radius=0.15).move_to(image_frame.get_center())
        self.play(image_content.animate.scale(1.2), run_time=1.5)
        self.play(FadeIn(target_icon), run_time=1)
        self.wait(1)

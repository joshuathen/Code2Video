from manim import *
import numpy as np

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
        # Setup the stage
        lecture_lines = [
            'The U-Net acts as a digital sculptor of noise.', 
            'It learns to predict the noise added at any step.', 
            'By subtracting this noise, we reveal the hidden structure.', 
            "The network doesn't draw; it iteratively removes chaos.", 
            'This reverse process breathes life back into the static.'
        ]
        self.setup_layout("The Reverse Process: The U-Net Sculptor", lecture_lines)
        
        # Colors
        UNET_COLOR = "#BB88FF"
        NOISE_COLOR = "#AAAAAA"
        MATH_COLOR = "#00FF00"
        IMAGE_COLOR = WHITE

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Noisy Image block
        noisy_img_rect = Rectangle(height=1.0, width=1.0, color=IMAGE_COLOR, fill_opacity=0.1)
        # Use simple dots to represent noise
        noise_dots_1 = VGroup(*[
            Dot(
                point=[np.random.uniform(-0.4, 0.4), np.random.uniform(-0.4, 0.4), 0], 
                radius=0.015, 
                color=NOISE_COLOR
            ) for _ in range(30)
        ])
        noisy_group = VGroup(noisy_img_rect, noise_dots_1)
        self.place_at_grid(noisy_group, "B2", scale_factor=0.8)
        noisy_label = Text("Noisy Image", font_size=14).next_to(noisy_group, UP, buff=0.1)

        # U-Net box - adjusted size to fit B3-C4 area nicely
        unet_box = Rectangle(height=1.4, width=1.4, color=UNET_COLOR, fill_opacity=0.3)
        unet_text = Text("U-Net", font_size=20, color=UNET_COLOR)
        unet_group = VGroup(unet_box, unet_text)
        self.place_in_area(unet_group, "B3", "C4", scale_factor=1.0)

        self.play(FadeIn(noisy_group), FadeIn(noisy_label), FadeIn(unet_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)

        # Arrow from Noisy Image into U-Net
        arrow_to_unet = Arrow(noisy_group.get_right(), unet_group.get_left(), buff=0.1, color=WHITE)
        
        # Output: Predicted Noise
        pred_noise_rect = Rectangle(height=1.0, width=1.0, color=NOISE_COLOR, fill_opacity=0.2)
        pred_noise_dots = VGroup(*[
            Dot(
                point=[np.random.uniform(-0.4, 0.4), np.random.uniform(-0.4, 0.4), 0], 
                radius=0.015, 
                color=NOISE_COLOR
            ) for _ in range(30)
        ])
        pred_noise_group = VGroup(pred_noise_rect, pred_noise_dots)
        self.place_at_grid(pred_noise_group, "B5", scale_factor=1.0)
        pred_label = Text("Noise Prediction", font_size=14, color=NOISE_COLOR).next_to(pred_noise_group, UP, buff=0.1)
        
        arrow_from_unet = Arrow(unet_group.get_right(), pred_noise_group.get_left(), buff=0.1, color=WHITE)

        # Math formula: p_theta(xt-1 | xt)
        p_char = Text("p", font_size=22, color=MATH_COLOR)
        theta_char = Text("θ", font_size=16, color=MATH_COLOR)
        theta_char.next_to(p_char.get_corner(DR), RIGHT, buff=0.02).shift(UP*0.05)
        open_paren = Text("(", font_size=22, color=MATH_COLOR).next_to(theta_char, RIGHT, buff=0.05)
        x_char1 = Text("x", font_size=22, color=MATH_COLOR).next_to(open_paren, RIGHT, buff=0.05)
        sub_t1 = Text("t-1", font_size=14, color=MATH_COLOR).next_to(x_char1.get_corner(DR), RIGHT, buff=0.02).shift(UP*0.05)
        pipe_char = Text("|", font_size=22, color=MATH_COLOR).next_to(sub_t1, RIGHT, buff=0.1)
        x_char2 = Text("x", font_size=22, color=MATH_COLOR).next_to(pipe_char, RIGHT, buff=0.1)
        sub_t2 = Text("t", font_size=14, color=MATH_COLOR).next_to(x_char2.get_corner(DR), RIGHT, buff=0.02).shift(UP*0.05)
        close_paren = Text(")", font_size=22, color=MATH_COLOR).next_to(sub_t2, RIGHT, buff=0.05)
        
        math_obj = VGroup(p_char, theta_char, open_paren, x_char1, sub_t1, pipe_char, x_char2, sub_t2, close_paren)
        self.place_at_grid(math_obj, "C5", scale_factor=0.9)

        self.play(GrowArrow(arrow_to_unet))
        self.play(GrowArrow(arrow_from_unet), FadeIn(pred_noise_group), FadeIn(pred_label))
        self.play(FadeIn(math_obj))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)

        # Subtraction: Image - Noise = Cleaner Image
        minus_op = Text("-", font_size=28)
        equal_op = Text("=", font_size=28)
        
        noisy_sub_copy = noisy_group.copy()
        pred_sub_copy = pred_noise_group.copy()
        
        clean_img_rect = Rectangle(height=1.0, width=1.0, color=IMAGE_COLOR, fill_opacity=0.1)
        # Fewer dots, suggesting cleanup
        clean_dots = VGroup(*[
            Dot(
                point=[np.random.uniform(-0.4, 0.4), np.random.uniform(-0.4, 0.4), 0], 
                radius=0.01, 
                color=WHITE
            ) for _ in range(8)
        ])
        clean_result = VGroup(clean_img_rect, clean_dots)
        
        self.place_at_grid(noisy_sub_copy, "D2", scale_factor=0.8)
        self.place_at_grid(minus_op, "D3", scale_factor=1.0)
        self.place_at_grid(pred_sub_copy, "D4", scale_factor=0.8)
        self.place_at_grid(equal_op, "D5", scale_factor=1.0)
        self.place_at_grid(clean_result, "D6", scale_factor=0.8)

        self.play(
            FadeIn(noisy_sub_copy), 
            FadeIn(minus_op), 
            FadeIn(pred_sub_copy), 
            FadeIn(equal_op), 
            FadeIn(clean_result)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)

        # Fade out previous step for clarity in new analogy
        self.play(
            FadeOut(noisy_sub_copy), 
            FadeOut(minus_op), 
            FadeOut(pred_sub_copy), 
            FadeOut(equal_op), 
            FadeOut(clean_result),
            FadeOut(noisy_group),
            FadeOut(noisy_label),
            FadeOut(unet_group),
            FadeOut(arrow_to_unet),
            FadeOut(arrow_from_unet),
            FadeOut(pred_noise_group),
            FadeOut(pred_label),
            FadeOut(math_obj)
        )
        
        # Sculptor Analogy: Marble block
        marble_block = Rectangle(height=1.5, width=2.5, color=NOISE_COLOR, fill_opacity=0.6)
        self.place_in_area(marble_block, "E2", "F4", scale_factor=1.0)
        marble_label = Text("Marble (Noise)", font_size=16).next_to(marble_block, UP, buff=0.1)
        
        # Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/dragon.svg
        dragon_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/dragon.svg")
        dragon_icon.set_color(WHITE)
        dragon_icon.scale(0.8)
        dragon_icon.move_to(marble_block.get_center())
        dragon_icon.set_opacity(0)

        self.play(FadeIn(marble_block), FadeIn(marble_label))
        
        # Chipping away process
        self.play(
            marble_block.animate.set_opacity(0.1),
            dragon_icon.animate.set_opacity(0.4),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Breathe life back: Dragon glows GOLD
        self.play(
            dragon_icon.animate.set_color(GOLD).set_opacity(1).scale(1.2)
        )
        self.wait(2)

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
        self.setup_layout("The Denoising Engine: Integrating CLIP Guidance", [
            "The denoiser predicts noise to subtract.",
            "CLIP vectors steer the generation process.",
            "We maximize similarity to the prompt.",
            "This creates a targeted visual output.",
            "Styles are refined through guided diffusion."
        ])
        
        # Elements
        camera_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/camera.svg").set_color(WHITE)
        canvas_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/canvas.svg").set_color(WHITE)
        
        clip_model = Rectangle(width=2, height=1, fill_opacity=0.5, color=BLUE)
        guidance_vec = Arrow(start=LEFT, end=RIGHT, color=GOLD)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FF0000")
        self.place_at_grid(camera_icon, 'C2', scale_factor=0.8)
        self.place_at_grid(clip_model, 'C5', scale_factor=0.8)
        self.play(FadeIn(camera_icon), FadeIn(clip_model))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FFFF")
        self.place_at_grid(guidance_vec, 'E3', scale_factor=0.7)
        self.play(GrowArrow(guidance_vec))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#0000FF")
        self.play(Indicate(guidance_vec))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FFFF00")
        self.play(camera_icon.animate.set_color(WHITE))
        self.wait(1)
        
        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#00FF00")
        self.place_at_grid(canvas_icon, 'C3', scale_factor=0.8)
        self.play(FadeOut(guidance_vec), FadeOut(clip_model), FadeOut(camera_icon), FadeIn(canvas_icon))
        self.wait(1)

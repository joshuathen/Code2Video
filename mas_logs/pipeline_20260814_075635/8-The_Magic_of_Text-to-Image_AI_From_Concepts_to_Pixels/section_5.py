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
        self.setup_layout("Summary & Future Outlook", [
            "CLIP and Diffusion power modern AI.",
            "Latent models improve generation efficiency.",
            "This bridge transforms ideas into art."
        ])

        # Define assets
        clip = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/camera.svg", color=WHITE)
        diff = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg", color=WHITE)
        guid = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/palette.svg", color=WHITE)
        
        self.place_at_grid(clip, 'B2', 0.5)
        self.place_at_grid(diff, 'B4', 0.5)
        self.place_at_grid(guid, 'B6', 0.5)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFCCCC"))
        self.play(FadeIn(clip), FadeIn(diff), FadeIn(guid))

        # === Animation for Lecture Line 2 ===
        line1 = Line(clip.get_right(), diff.get_left(), color="#00FF00")
        line2 = Line(diff.get_right(), guid.get_left(), color="#00FF00")
        
        self.play(self.lecture[1].animate.set_color("#CCFFCC"))
        self.play(Create(line1), Create(line2))
        
        # Pulsing effect via updater
        pulsing_val = ValueTracker(0)
        line1.add_updater(lambda m: m.set_stroke(opacity=0.5 + 0.5 * np.sin(pulsing_val.get_value())))
        line2.add_updater(lambda m: m.set_stroke(opacity=0.5 + 0.5 * np.sin(pulsing_val.get_value())))
        self.play(pulsing_val.animate.set_value(2 * PI), run_time=2)
        line1.remove_updater(line1.updaters[0])
        line2.remove_updater(line2.updaters[0])

        # === Animation for Lecture Line 3 ===
        future_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/paintbrush.svg", color="#FFD700")
        future_text = Text("Future: Unified Generative AI", color="#FFD700").scale(0.8)
        self.place_at_grid(future_icon, 'E3', 0.5)
        self.place_in_area(future_text, 'F1', 'F6', scale_factor=0.6)
        
        self.play(self.lecture[2].animate.set_color("#CCCCFF"))
        self.play(
            FadeOut(clip), FadeOut(diff), FadeOut(guid),
            FadeOut(line1), FadeOut(line2),
            FadeIn(future_icon), FadeIn(future_text)
        )
        self.wait(2)

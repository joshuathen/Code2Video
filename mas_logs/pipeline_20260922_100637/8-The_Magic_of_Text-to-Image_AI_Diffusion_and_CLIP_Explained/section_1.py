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

class Section1Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Introduction: The Creative Loop", [
            "Text-to-image AI bridges language and visual space.",
            "CLIP acts as our knowledgeable creative guide.",
            "Diffusion models function as our talented artist.",
            "Together they turn prompts into beautiful images."
        ])
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00FFFF")
        text_label = Text("Creative Loop", font_size=36, color=WHITE)
        self.place_at_grid(text_label, 'B3', scale_factor=0.9)
        self.play(Write(text_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#33A1FF")
        clip_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/brain.svg", color="#33A1FF")
        self.place_at_grid(clip_icon, 'B5', scale_factor=0.6)
        clip_label = Text("CLIP", font_size=24, color="#33A1FF")
        clip_label.next_to(clip_icon, DOWN, buff=0.1)
        self.play(FadeIn(clip_icon), Write(clip_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FF33A1")
        diffusion_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/paintbrush.svg", color="#FF33A1")
        self.place_at_grid(diffusion_icon, 'D5', scale_factor=0.6)
        diffusion_label = Text("Diffusion", font_size=24, color="#FF33A1")
        diffusion_label.next_to(diffusion_icon, DOWN, buff=0.1)
        self.play(FadeIn(diffusion_icon), Write(diffusion_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color("#FFFF33")
        cycle_connection = CurvedArrow(clip_icon.get_bottom(), diffusion_icon.get_top(), color="#FFFF33")
        self.play(Create(cycle_connection))
        self.wait(2)

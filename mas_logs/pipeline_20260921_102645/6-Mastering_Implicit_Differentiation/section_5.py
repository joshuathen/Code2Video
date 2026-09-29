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
        self.setup_layout("Summary & Pro-Tips", [
            "Always apply Chain Rule to 'y' terms.",
            "Keep dy/dx terms on one side.",
            "If you see 'y', append dy/dx."
        ])
        
        # Assets
        asset_scale = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/scale.svg")
        asset_magnet = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/magnet.svg")
        asset_anchor = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/anchor.svg")

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        bullet1 = Text("Differentiate all terms.", font_size=20, color=YELLOW)
        self.place_at_grid(bullet1, 'B4', scale_factor=0.8)
        self.place_at_grid(asset_scale, 'B2', scale_factor=0.5)
        self.play(FadeIn(bullet1), FadeIn(asset_scale))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(BLUE))
        bullet2 = Text("Isolate dy/dx terms.", font_size=20, color=BLUE)
        self.place_at_grid(bullet2, 'C4', scale_factor=0.8)
        self.place_at_grid(asset_magnet, 'C2', scale_factor=0.5)
        self.play(FadeIn(bullet2), FadeIn(asset_magnet))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(RED))
        bullet3 = Text("Always check implicit consistency.", font_size=20, color=RED)
        self.place_at_grid(bullet3, 'D4', scale_factor=0.8)
        self.place_at_grid(asset_anchor, 'D2', scale_factor=0.5)
        self.play(FadeIn(bullet3), FadeIn(asset_anchor))

        self.wait(2)

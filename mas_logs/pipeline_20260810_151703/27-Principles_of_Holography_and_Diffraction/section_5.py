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
        self.setup_layout("Summary & Application", [
            "Holograms are diffraction patterns stored on media.", 
            "They enable secure data storage and advanced imaging.", 
            "Holographic displays project information onto real-world views."
        ])
        
        # Assets
        workspace_img = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/workspace.svg")
        hologram_img = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/hologram.svg")
        card_img = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/card.png")
        headset_img = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/headset.svg")
        
        # Security sticker group (using Group because ImageMobject is not a VMobject)
        security_group = Group(hologram_img, card_img).arrange(DOWN)

        # === Animation for Lecture Line 1 ===
        self.place_at_grid(workspace_img, 'C6', scale_factor=0.6)
        self.play(Create(workspace_img), self.lecture[0].animate.set_color("#00FFFF"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.place_in_area(security_group, 'D5', 'D6', scale_factor=0.7)
        ray = Line(start=self.grid['D3'], end=self.grid['D6'], color=WHITE)
        self.play(FadeIn(security_group), Create(ray), self.lecture[1].animate.set_color("#FF00FF"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.place_at_grid(headset_img, 'E5', scale_factor=0.5)
        self.play(GrowFromCenter(headset_img), self.lecture[2].animate.set_color("#FFFF00"))
        self.wait(1)

        # Final fade
        self.play(FadeOut(self.lecture), FadeOut(self.title), FadeOut(workspace_img), FadeOut(security_group), FadeOut(ray), FadeOut(headset_img))

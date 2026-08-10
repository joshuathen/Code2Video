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
        self.setup_layout("The Problem: Context Blindness", [
            "Context is hard for simple models to maintain.",
            "'The bank' needs 'river' to be understood.",
            "Old models forget context over long distances."
        ])
        
        # Load SVG Assets
        bank_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bank.svg")
        river_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/river.svg")
        
        query_label = Text("Query", font_size=24, color=WHITE)
        key_label = Text("Key", font_size=24, color=WHITE)
        value_label = Text("Value", font_size=24, color=WHITE)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF5733"))
        self.place_at_grid(query_label, "B2", scale_factor=0.5)
        self.place_at_grid(bank_icon, "B3", scale_factor=0.6)
        self.play(FadeIn(query_label), FadeIn(bank_icon))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#33FF57"))
        self.place_at_grid(key_label, "B5", scale_factor=0.5)
        self.place_at_grid(river_icon, "B4", scale_factor=0.6)
        
        self.play(FadeIn(key_label), FadeIn(river_icon))
        self.play(Indicate(bank_icon, color="#FF5733"), Indicate(river_icon, color="#33FF57"))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#3357FF"))
        self.place_at_grid(value_label, "C5", scale_factor=0.5)
        
        vector_group = VGroup(bank_icon, river_icon, query_label, key_label, value_label)
        self.place_in_area(vector_group, "B3", "D5", scale_factor=0.7)
        self.play(FadeIn(value_label))
        
        self.wait(1)

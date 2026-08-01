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
        # 1. Setup layout
        self.setup_layout("The Comparison Matrix (Graphical Analysis)", [
            "Let's compare an orange to a true nut.",
            "Oranges are juicy, while nuts are completely dry.",
            "Oranges have leathery skins; nuts have stony shells.",
            "Oranges have many seeds; nuts have only one.",
            "Scientifically, an orange is definitely not a nut!"
        ])

        # === Animation for Lecture Line 1 ===
        # A comparison table appears with columns for 'Orange' and 'True Nut'.
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        header_orange = Text("Orange", font_size=24, color="#FFA500")
        header_nut = Text("True Nut", font_size=24, color="#D2B48C")
        
        self.place_at_grid(header_orange, "A3")
        self.place_at_grid(header_nut, "A5")
        
        self.play(FadeIn(header_orange), FadeIn(header_nut))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # The 'Moisture' row displays 'Juicy' (#FFA500) versus 'Dry' (#D2B48C).
        self.play(self.lecture[1].animate.set_color("#FFA500"))
        
        label_moisture = Text("Moisture", font_size=20, color=WHITE)
        val_juicy = Text("Juicy", font_size=20, color="#FFA500")
        val_dry = Text("Dry", font_size=20, color="#D2B48C")
        
        self.place_at_grid(label_moisture, "B1")
        self.place_at_grid(val_juicy, "B3")
        self.place_at_grid(val_dry, "B5")
        
        self.play(Write(label_moisture), Write(val_juicy), Write(val_dry))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The 'Pericarp' row displays 'Leathery' (#FFA500) versus 'Stony' (#D2B48C).
        self.play(self.lecture[2].animate.set_color("#FFA500"))
        
        label_pericarp = Text("Pericarp", font_size=20, color=WHITE)
        val_leathery = Text("Leathery", font_size=20, color="#FFA500")
        val_stony = Text("Stony", font_size=20, color="#D2B48C")
        
        self.place_at_grid(label_pericarp, "C1")
        self.place_at_grid(val_leathery, "C3")
        self.place_at_grid(val_stony, "C5")
        
        self.play(Write(label_pericarp), Write(val_leathery), Write(val_stony))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # The 'Seeds' row displays 'Many' (#FFA500) versus 'One' (#D2B48C).
        self.play(self.lecture[3].animate.set_color("#FFA500"))
        
        label_seeds = Text("Seeds", font_size=20, color=WHITE)
        val_many = Text("Many", font_size=20, color="#FFA500")
        val_one = Text("One", font_size=20, color="#D2B48C")
        
        self.place_at_grid(label_seeds, "D1")
        self.place_at_grid(val_many, "D3")
        self.place_at_grid(val_one, "D5")
        
        self.play(Write(label_seeds), Write(val_many), Write(val_one))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # A large red 'X' (#FF0000) appears over the comparison to show they differ.
        self.play(self.lecture[4].animate.set_color("#FF0000"))
        
        red_x = Text("X", font_size=120, color="#FF0000", weight=BOLD)
        self.place_in_area(red_x, "A2", "D6")
        
        self.play(Write(red_x))
        self.wait(2)

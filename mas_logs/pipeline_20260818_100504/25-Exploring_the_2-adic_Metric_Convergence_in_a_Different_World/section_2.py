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

class Section2Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["The 2-adic norm measures divisibility by two.", "Higher powers of two mean smaller values.", "Eight is small; seven is large."]
        self.setup_layout("Introducing the 2-adic Norm", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        integers = VGroup(*[Text(str(i), font_size=24) for i in range(1, 9)])
        integers.arrange(RIGHT, buff=0.2).scale(0.8)
        
        # Integrate Asset - SVGMobject is required for .svg files, ImageMobject is for raster formats
        seven_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/seven.svg")
        seven_icon.scale(0.1)
        
        self.place_in_area(integers, "A1", "B6", scale_factor=0.8)
        self.play(FadeIn(integers), FadeIn(seven_icon))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF00FF")
        # Visualizing 8 = 2^3
        decomp = Text("8 = 2^3", font_size=32, color="#FF00FF")
        self.place_at_grid(decomp, "D2", scale_factor=0.9)
        self.play(FadeIn(decomp))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FF00")
        
        # Integrate Asset - SVGMobject is required for .svg files
        eight_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/eight.svg")
        eight_icon.scale(0.1)
        
        # Bar chart idea: 8 is small (low height), 7 is large (high height)
        bar_8 = Rectangle(width=0.4, height=0.5, color="#00FF00", fill_opacity=1)
        label_8 = Text("8", font_size=20).next_to(bar_8, DOWN)
        
        bar_7 = Rectangle(width=0.4, height=2.0, color="#FF0000", fill_opacity=1)
        label_7 = Text("7", font_size=20).next_to(bar_7, DOWN)
        
        chart = VGroup(bar_8, label_8, bar_7, label_7).arrange(RIGHT, buff=0.5)
        self.place_in_area(chart, "D4", "F6", scale_factor=0.7)
        
        self.play(Create(bar_8), Write(label_8), Create(bar_7), Write(label_7), FadeIn(eight_icon))
        self.wait(2)

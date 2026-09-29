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

class Section3Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Step 1: Ephemeral Identifier Generation", [
            "Daily keys generate ephemeral identifiers.",
            "IDs rotate every fifteen minutes.",
            "Observed IDs appear random and disconnected."
        ])
        
        # Asset paths
        key_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/key.svg"
        gen_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/generator.svg"
        tl_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/timeline.svg"

        # === Animation for Lecture Line 1 ===
        dtk = SVGMobject(key_asset, color="#EFC050")
        generator = SVGMobject(gen_asset, color="#6B5B95")
        gen_label = Text("Generator", color="#6B5B95", font_size=20)
        generator_group = VGroup(generator, gen_label).arrange(DOWN)
        
        self.place_in_area(dtk, 'D2', 'D3', scale_factor=1.0)
        self.place_at_grid(generator_group, 'D5', scale_factor=0.9)
        self.play(FadeIn(dtk), FadeIn(generator_group))
        self.lecture[0].set_color("#EFC050")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF6F61")
        
        timeline = SVGMobject(tl_asset)
        self.place_in_area(timeline, 'C1', 'C6', scale_factor=1.5)
        
        t1 = Text("10:00", font_size=18).move_to(self.grid['D1'])
        t2 = Text("10:15", font_size=18).move_to(self.grid['D3'])
        t3 = Text("10:30", font_size=18).move_to(self.grid['D5'])
        
        id1 = Text("ID_1", color="#FF6F61", font_size=20).move_to(self.grid['C1'])
        id2 = Text("ID_2", color="#88B04B", font_size=20).move_to(self.grid['C3'])
        id3 = Text("ID_3", color="#D64161", font_size=20).move_to(self.grid['C5'])
        
        self.play(Create(timeline), Write(VGroup(t1, t2, t3)))
        self.play(FadeIn(id1), FadeIn(id2), FadeIn(id3))
        self.lecture[1].set_color("#88B04B")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFFFF")
        
        link1 = Line(dtk.get_bottom(), id1.get_top(), color="#FFFFFF", stroke_width=2)
        link2 = Line(dtk.get_bottom(), id2.get_top(), color="#FFFFFF", stroke_width=2)
        link3 = Line(dtk.get_bottom(), id3.get_top(), color="#FFFFFF", stroke_width=2)
        
        self.play(Create(link1), Create(link2), Create(link3))
        self.wait(2)

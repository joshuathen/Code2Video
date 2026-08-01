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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout(
            "Prerequisite: Base-3 (Ternary) Counting", 
            [
                'Ternary counting uses digits zero, one, and two.', 
                'Watch the counter increment in base-three logic.', 
                'Each digit maps to a specific peg: A, B, or C.'
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#87CEEB"))
        
        dec_label = Text("Decimal: 0-9", font_size=24, color=WHITE)
        bin_label = Text("Binary: 0-1", font_size=24, color=WHITE)
        
        self.place_in_area(dec_label, "B1", "B3", scale_factor=0.8)
        self.place_in_area(bin_label, "B4", "B6", scale_factor=0.8)
        
        tern_label = Text("Ternary: 0, 1, 2", font_size=32, color="#87CEEB")
        self.place_in_area(tern_label, "C2", "C5", scale_factor=0.8)
        
        self.play(FadeIn(dec_label), FadeIn(bin_label))
        self.play(FadeIn(tern_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFD700"))
        
        # Odometer Assets
        disk_path = "/mmfs1/data/home/jthen/Code2Video/assets/icon/disk.svg"
        disk1 = SVGMobject(disk_path, color=WHITE).scale(0.3)
        disk2 = SVGMobject(disk_path, color=WHITE).scale(0.2)
        disk3 = SVGMobject(disk_path, color=WHITE).scale(0.1)
        
        d1 = Text("0", font_size=40, color=WHITE)
        d2 = Text("0", font_size=40, color=WHITE)
        d3 = Text("0", font_size=40, color=WHITE)
        
        # Group disks with digits
        col1 = VGroup(disk1, d1).arrange(DOWN, buff=0.2)
        col2 = VGroup(disk2, d2).arrange(DOWN, buff=0.2)
        col3 = VGroup(disk3, d3).arrange(DOWN, buff=0.2)
        
        odometer_group = VGroup(col1, col2, col3).arrange(RIGHT, buff=0.8)
        self.place_in_area(odometer_group, "D1", "D6", scale_factor=1.0)
        
        box = Rectangle(height=1.5, width=4.5, color="#FFD700")
        box.move_to(odometer_group.get_center())
        
        self.play(Create(box), FadeIn(odometer_group))
        self.wait(1)
        
        # Increment 000 -> 001
        new_d3_1 = Text("1", font_size=40, color=WHITE).move_to(d3)
        self.play(ReplacementTransform(d3, new_d3_1))
        self.wait(0.5)
        
        # Increment 001 -> 002
        new_d3_2 = Text("2", font_size=40, color=WHITE).move_to(new_d3_1)
        self.play(ReplacementTransform(new_d3_1, new_d3_2))
        self.wait(0.5)
        
        # Increment 002 -> 010
        new_d2_1 = Text("1", font_size=40, color=WHITE).move_to(d2)
        new_d3_0 = Text("0", font_size=40, color=WHITE).move_to(new_d3_2)
        self.play(
            ReplacementTransform(d2, new_d2_1),
            ReplacementTransform(new_d3_2, new_d3_0)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#90EE90"))
        
        # Peg labels and Assets
        peg_path = "/mmfs1/data/home/jthen/Code2Video/assets/icon/peg.svg"
        
        pegA_icon = SVGMobject(peg_path, color="#FFFFFF").scale(0.2)
        pegB_icon = SVGMobject(peg_path, color="#87CEEB").scale(0.2)
        pegC_icon = SVGMobject(peg_path, color="#90EE90").scale(0.2)
        
        peg_legend = VGroup(
            VGroup(pegA_icon, Text("0 = Peg A", font_size=20, color="#FFFFFF")).arrange(RIGHT, buff=0.1),
            VGroup(pegB_icon, Text("1 = Peg B", font_size=20, color="#87CEEB")).arrange(RIGHT, buff=0.1),
            VGroup(pegC_icon, Text("2 = Peg C", font_size=20, color="#90EE90")).arrange(RIGHT, buff=0.1)
        ).arrange(RIGHT, buff=0.4)
        
        self.place_in_area(peg_legend, "E1", "E6", scale_factor=0.7)
        
        # Color current digits in odometer based on mapping
        self.play(
            d1.animate.set_color("#FFFFFF"),
            new_d2_1.animate.set_color("#87CEEB"),
            new_d3_0.animate.set_color("#FFFFFF"),
            FadeIn(peg_legend)
        )
        self.wait(1)
        
        # Final calculation visual: 111 ternary to 13 decimal
        final_d1 = Text("1", font_size=40, color="#87CEEB").move_to(d1)
        final_d2 = Text("1", font_size=40, color="#87CEEB").move_to(new_d2_1)
        final_d3 = Text("1", font_size=40, color="#87CEEB").move_to(new_d3_0)
        
        calc_text = Text("(1 x 9) + (1 x 3) + (1 x 1) = 13", font_size=24, color="#90EE90")
        self.place_in_area(calc_text, "F1", "F6", scale_factor=0.8)
        
        self.play(
            ReplacementTransform(d1, final_d1),
            ReplacementTransform(new_d2_1, final_d2),
            ReplacementTransform(new_d3_0, final_d3),
            Write(calc_text)
        )
        self.wait(2)

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

class Section1Scene(TeachingScene):
    def construct(self):
        title = "The Pizza Slicing Challenge"
        lines = [
            "Max the spider wants to divide his circular web.",
            "Two points connected by a chord create two regions.",
            "Three points create four regions inside the circle.",
            "Four points create eight regions as connections grow.",
            "Five points create sixteen regions, doubling the count."
        ]
        self.setup_layout(title, lines)

        # Helper for circle geometry
        # Issue 35: Expanded grid area and increased scale factor
        main_circle = Circle(radius=1.8, color="#FFFFFF", stroke_width=4)
        self.place_in_area(main_circle, 'A2', 'E5', scale_factor=1.1)
        
        circle_center = main_circle.get_center()
        circle_radius = 1.8 * 1.1

        def get_circle_pt(angle_deg):
            return circle_center + np.array([
                circle_radius * np.cos(angle_deg * DEGREES),
                circle_radius * np.sin(angle_deg * DEGREES),
                0
            ])

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Issue 31: Asset Integration for Max the Spider
        max_spider_svg = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/spider.svg", color=WHITE)
        max_label = Text("Max", font_size=16, color=WHITE)
        max_spider = VGroup(max_spider_svg, max_label).arrange(UP, buff=0.1).scale(0.3)
        
        # Place spider at the top edge of the circle
        max_pos = get_circle_pt(90) + UP * 0.25
        max_spider.move_to(max_pos)

        self.play(Create(main_circle))
        self.play(FadeIn(max_spider))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)

        # --- N = 2 ---
        p2_angles = [90, 270]
        pts2 = [get_circle_pt(a) for a in p2_angles]
        dots2 = VGroup(*[Dot(p, color="#FFFFFF") for p in pts2])
        chord2 = Line(pts2[0], pts2[1], color="#0000FF")
        
        self.play(Create(dots2))
        self.play(Create(chord2))
        
        # Highlight regions for n=2
        region_labels2 = VGroup(
            Text("1", font_size=20).move_to(circle_center + LEFT*0.6),
            Text("2", font_size=20).move_to(circle_center + RIGHT*0.6)
        )
        self.play(LaggedStart(*[FadeIn(l) for l in region_labels2], lag_ratio=0.5))
        self.wait(1)
        self.play(FadeOut(dots2, chord2, region_labels2))

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)

        # --- N = 3 ---
        p3_angles = [90, 210, 330]
        pts3 = [get_circle_pt(a) for a in p3_angles]
        dots3 = VGroup(*[Dot(p, color="#FFFFFF") for p in pts3])
        chords3 = VGroup(
            Line(pts3[0], pts3[1], color="#0000FF"),
            Line(pts3[1], pts3[2], color="#0000FF"),
            Line(pts3[2], pts3[0], color="#0000FF")
        )
        
        self.play(Create(dots3))
        self.play(Create(chords3))
        
        # Highlight 4 regions
        region_centers3 = [
            circle_center + UP*0.5,
            circle_center + DOWN*0.4 + LEFT*0.5,
            circle_center + DOWN*0.4 + RIGHT*0.5,
            circle_center # center small triangle
        ]
        region_labels3 = VGroup(*[
            Text(str(i+1), font_size=20, color="#00FF00").move_to(pos)
            for i, pos in enumerate(region_centers3)
        ])
        self.play(LaggedStart(*[Write(l) for l in region_labels3], lag_ratio=0.4))
        self.wait(1)
        self.play(FadeOut(dots3, chords3, region_labels3))

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)

        # --- N = 4 ---
        p4_angles = [45, 135, 225, 315]
        pts4 = [get_circle_pt(a) for a in p4_angles]
        dots4 = VGroup(*[Dot(p, color="#FFFFFF") for p in pts4])
        chords4 = VGroup()
        for i in range(4):
            for j in range(i+1, 4):
                chords4.add(Line(pts4[i], pts4[j], color="#0000FF"))
        
        self.play(Create(dots4), Create(chords4))
        
        # 8 regions
        region_labels4 = VGroup(*[
            Text(str(i+1), font_size=18, color="#ADD8E6").move_to(
                circle_center + 1.1 * np.array([np.cos(a*DEGREES), np.sin(a*DEGREES), 0])
            ) for i, a in enumerate(np.linspace(0, 315, 8))
        ])
        self.play(LaggedStart(*[FadeIn(l) for l in region_labels4], lag_ratio=0.2))
        self.wait(1)
        self.play(FadeOut(dots4, chords4, region_labels4))

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)

        # --- N = 5 ---
        p5_angles = [90, 162, 234, 306, 18]
        pts5 = [get_circle_pt(a) for a in p5_angles]
        dots5 = VGroup(*[Dot(p, color="#FFFFFF") for p in pts5])
        chords5 = VGroup()
        for i in range(5):
            for j in range(i+1, 5):
                chords5.add(Line(pts5[i], pts5[j], color="#0000FF"))
        
        self.play(Create(dots5), Create(chords5))
        
        # 16 regions
        region_labels5 = VGroup(*[
            Text(str(i+1), font_size=16, color="#FFA500").move_to(
                circle_center + (0.7 if i < 10 else 1.4) * 
                np.array([np.cos(i*22.5*DEGREES), np.sin(i*22.5*DEGREES), 0])
            ) for i in range(16)
        ])
        self.play(LaggedStart(*[FadeIn(l) for l in region_labels5], lag_ratio=0.1))
        self.wait(1)

        # Issue 34: Improved count sequence positioning
        count_seq = Text("Count: 1, 2, 4, 8, 16", font_size=28, color="#FFA500")
        self.place_in_area(count_seq, 'F2', 'F5', scale_factor=0.8)
        self.play(Write(count_seq))
        self.wait(1)

        # hook: Many chords and question mark
        q_mark = Text("?", font_size=120, color="#FFFFFF").move_to(circle_center)
        random_chords = VGroup()
        for _ in range(15):
            a1, a2 = np.random.uniform(0, 360, 2)
            random_chords.add(Line(get_circle_pt(a1), get_circle_pt(a2), color="#0000FF", stroke_opacity=0.4))
        
        self.play(
            FadeOut(dots5, chords5, region_labels5),
            FadeIn(random_chords),
            Write(q_mark)
        )
        self.wait(2)

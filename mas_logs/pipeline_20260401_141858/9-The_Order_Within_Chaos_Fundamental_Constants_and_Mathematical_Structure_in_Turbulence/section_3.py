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

class Section3Scene(TeachingScene):
    def construct(self):
        # Mandatory lecture lines
        lecture_lines = [
            'Energy enters through large, macroscopic fluid structures.', 
            'Large eddies break down, feeding energy into smaller ones.', 
            'Tiny yellow scales represent the end of the cascade.'
        ]
        
        self.setup_layout("The Energy Cascade: Richardson’s Vision", lecture_lines)

        # Colors
        BLUE_WHORL = "#58C4DD"
        GREEN_WHORL = "#83C167"
        YELLOW_WHORL = "#F8E71C"
        
        # Asset Path
        the_svg_path = "/mmfs1/data/home/jthen/Code2Video/assets/icon/the.svg"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE_WHORL))
        
        large_circle = Circle(radius=1.5, color=BLUE_WHORL, stroke_width=4)
        dashes = DashedVMobject(Circle(radius=1.2, color=BLUE_WHORL), num_dashes=15)
        large_whorl = VGroup(large_circle, dashes)
        # Issue 33 Fix: Place in B3-E5 area
        self.place_in_area(large_whorl, 'B3', 'E5', scale_factor=1.0)
        
        # [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/the.svg] used for transition indicator
        asset_1 = SVGMobject(the_svg_path).scale(0.3).set_color(WHITE)
        self.place_at_grid(asset_1, 'D4')
        
        self.play(Create(large_whorl), FadeIn(asset_1))
        self.play(Rotate(large_whorl, angle=2*PI, run_time=3, rate_func=linear))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(GREEN_WHORL))
        
        # Medium whorls
        med_whorl_1 = VGroup(Circle(radius=0.6, color=GREEN_WHORL), DashedVMobject(Circle(radius=0.45, color=GREEN_WHORL), num_dashes=10))
        med_whorl_2 = VGroup(Circle(radius=0.6, color=GREEN_WHORL), DashedVMobject(Circle(radius=0.45, color=GREEN_WHORL), num_dashes=10))
        
        # Issue 34 Fix: Grid B2, scale 0.6
        self.place_at_grid(med_whorl_1, 'B2', scale_factor=0.6)
        # Issue 35 Fix: Grid E6, scale 0.6
        self.place_at_grid(med_whorl_2, 'E6', scale_factor=0.6)

        # [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/the.svg] used for comparison indicator
        asset_2 = SVGMobject(the_svg_path).scale(0.3).set_color(WHITE)
        self.place_at_grid(asset_2, 'A2')

        self.play(
            large_whorl.animate.set_stroke(opacity=0.3),
            FadeIn(med_whorl_1),
            FadeIn(med_whorl_2),
            FadeIn(asset_2)
        )
        
        # Medium whorls spin faster
        self.play(
            Rotate(med_whorl_1, angle=4*PI, run_time=2, rate_func=linear),
            Rotate(med_whorl_2, angle=4*PI, run_time=2, rate_func=linear),
            Rotate(large_whorl, angle=PI, run_time=2, rate_func=linear)
        )

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW_WHORL))
        
        small_whorls = VGroup()
        positions = ["A1", "A3", "C1", "D6", "F4", "F6"]
        for pos in positions:
            sw = Circle(radius=0.2, color=YELLOW_WHORL, stroke_width=2)
            self.place_at_grid(sw, pos)
            small_whorls.add(sw)
            
        # [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/the.svg] used for smallest scale indicator
        asset_3 = SVGMobject(the_svg_path).scale(0.3).set_color(WHITE)
        self.place_at_grid(asset_3, 'C4')
            
        self.play(FadeIn(small_whorls), FadeIn(asset_3))
        
        # Dissipation representation
        dots = VGroup(*[Dot(point=self.grid[f"{r}{c}"], radius=0.02, color=YELLOW_WHORL) 
                        for r in ["A", "B", "C", "D", "E", "F"] 
                        for c in ["1", "2", "3", "4", "5", "6"]])
        
        self.play(
            *[Rotate(sw, angle=8*PI, run_time=3, rate_func=linear) for sw in small_whorls],
            Rotate(med_whorl_1, angle=4*PI, run_time=3, rate_func=linear),
            Rotate(med_whorl_2, angle=4*PI, run_time=3, rate_func=linear),
            FadeIn(dots, lag_ratio=0.05)
        )
        
        # Cleanup
        self.play(
            FadeOut(large_whorl),
            FadeOut(med_whorl_1),
            FadeOut(med_whorl_2),
            FadeOut(small_whorls),
            FadeOut(dots),
            FadeOut(asset_1),
            FadeOut(asset_2),
            FadeOut(asset_3),
            run_time=2
        )
        self.wait(2)

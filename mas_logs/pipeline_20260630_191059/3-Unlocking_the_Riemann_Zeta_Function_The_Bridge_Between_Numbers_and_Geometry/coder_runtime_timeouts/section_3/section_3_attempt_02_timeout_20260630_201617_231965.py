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
        title = "Defining the Zeta Function"
        lines = [
            "The zeta function sums one over n to power s.",
            "At s equals one, the harmonic series explodes infinitely.",
            "As s increases, the sum settles into finite values.",
            "For s equals two, we solve the famous Basel Problem.",
            "The function converges for all values greater than one."
        ]
        self.setup_layout(title, lines)

        # Asset Paths
        tower_path = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/tower.svg"
        scale_path = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/scale.svg"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(WHITE))
        # Formula using Text for stability across environments
        zeta_formula = Text("ζ(s) = Σ 1/n^s", font_size=36, color=WHITE)
        self.place_in_area(zeta_formula, 'A2', 'B5')
        self.play(Write(zeta_formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(BLUE_B))
        
        s_tracker = ValueTracker(2.0)
        
        # Slider setup
        slider_line = NumberLine(x_range=[1, 4, 1], length=3, color=GRAY, include_numbers=True, font_size=18, label_constructor=Text)
        slider_label = Text("s =", font_size=24)
        # Use Text-based DecimalNumber for performance
        s_val_text = DecimalNumber(s_tracker.get_value(), num_decimal_places=1, font_size=24, mob_class=Text)
        s_val_text.add_updater(lambda d: d.set_value(s_tracker.get_value()))
        
        slider_group = VGroup(slider_label, s_val_text, slider_line).arrange(RIGHT, buff=0.2)
        self.place_at_grid(slider_group, 'C3', scale_factor=0.8)
        
        slider_dot = Dot(color=YELLOW)
        slider_dot.add_updater(lambda d: d.move_to(slider_line.n2p(s_tracker.get_value())))
        
        # Tower setup with asset
        tower_icon = SVGMobject(tower_path, color=BLUE_B, fill_opacity=0.2)
        self.place_at_grid(tower_icon, 'F3', scale_factor=1.0)
        
        # Tower terms (rectangles)
        rects = VGroup(*[Rectangle(width=0.4, height=0.1, fill_opacity=0.8, color=BLUE_B) for _ in range(8)])
        
        def update_tower(m):
            s_val = s_tracker.get_value()
            curr_pos = tower_icon.get_bottom()
            for i, r in enumerate(m):
                n = i + 1
                # Scaling factor to keep tower within screen at s=1
                h = 1.8 / (n**s_val)
                r.set_height(h, stretch=True)
                r.move_to(curr_pos + UP * (h/2))
                curr_pos += UP * h
                
        rects.add_updater(update_tower)
        
        self.add(slider_dot, tower_icon, rects)
        self.play(FadeIn(slider_group), FadeIn(tower_icon), FadeIn(rects))
        self.play(s_tracker.animate.set_value(1.0), run_time=1.5)
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(CYAN))
        self.play(s_tracker.animate.set_value(2.5), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FFD700"))
        
        # Transition from tower to scale
        self.play(
            FadeOut(slider_group), 
            FadeOut(slider_dot), 
            FadeOut(tower_icon), 
            FadeOut(rects),
            run_time=0.8
        )
        
        # Balance Scale Construction with asset
        scale_icon = SVGMobject(scale_path, color=GRAY)
        self.place_in_area(scale_icon, 'D2', 'F5', scale_factor=1.2)
        
        # Squares for Basel Problem visual
        squares = VGroup(*[
            Square(side_length=0.7/(n**1), fill_opacity=0.7, color=CYAN) 
            for n in range(1, 6)
        ]).arrange(RIGHT, buff=0.05, aligned_edge=DOWN)
        
        # Approximate placement on scale pans
        squares.move_to(scale_icon.get_center() + LEFT * 0.9 + UP * 0.4)
        
        # Solution highlight
        basel_res_val = Text("π²/6", color="#FFD700", font_size=30)
        basel_res_val.move_to(scale_icon.get_center() + RIGHT * 0.9 + UP * 0.4)
        
        basel_full = Text("ζ(2) = π²/6", color="#FFD700", font_size=36)
        self.place_at_grid(basel_full, 'E6', scale_factor=1.0)
        
        self.play(FadeIn(scale_icon), FadeIn(squares), FadeIn(basel_res_val))
        self.play(Write(basel_full))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#90EE90"))
        
        conv_text = Text("Convergence: s > 1", font_size=24, color="#90EE90")
        conv_box = SurroundingRectangle(conv_text, color="#90EE90", buff=0.2)
        conv_group = VGroup(conv_box, conv_text)
        
        self.place_at_grid(conv_group, 'A6', scale_factor=1.0)
        self.play(FadeIn(conv_group))
        self.wait(2)
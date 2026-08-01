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

class Section6Scene(TeachingScene):
    def construct(self):
        # Updated lecture lines per prompt requirements
        lecture_lines = [
            'Let\'s compare "FUZZY" against the top starter "CRANE".',
            'Both start with over two thousand possible word candidates.',
            '"CRANE" shatters the list, while "FUZZY" barely reduces it.'
        ]
        self.setup_layout("The Simulation: Why 'CRANE' Wins", lecture_lines)

        # Colors
        COLOR_CRANE = "#2ECC71"  # Green
        COLOR_FUZZY = "#E74C3C"  # Red
        COLOR_HIGHLIGHT = "#F1C40F" # Yellow

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_HIGHLIGHT)
        
        # Issue 34: Asset Integration
        crane_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/crane.svg")
        crane_icon.set_color(COLOR_CRANE).set_height(0.6)
        
        # Fixes from Issue 53 for positions and scaling
        word_crane_text = Text('"CRANE"', color=WHITE, weight=BOLD)
        group_crane_header = VGroup(crane_icon, word_crane_text).arrange(RIGHT, buff=0.2)
        # Issue 53 Fix 1: CRANE at A3
        self.place_at_grid(group_crane_header, "A3", scale_factor=0.8)
        
        # Issue 53 Fix 3: FUZZY at A5
        word_fuzzy = Text('"FUZZY"', color=WHITE, weight=BOLD)
        self.place_at_grid(word_fuzzy, "A5", scale_factor=0.8)
        
        # Vertical divider to "split the screen" on the animation side
        divider = Line(self.grid["A4"] + UP*0.5, self.grid["F4"] + DOWN*0.5, color=GREY, stroke_width=2)
        
        self.play(
            FadeIn(group_crane_header),
            FadeIn(word_fuzzy),
            Create(divider)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_HIGHLIGHT)
        
        # Suspect counters
        label_crane = Text("Suspects:", font_size=24, color=WHITE)
        num_crane = Text("2,300", font_size=24, color=WHITE)
        group_crane_count = VGroup(label_crane, num_crane).arrange(DOWN, buff=0.1)
        
        label_fuzzy = Text("Suspects:", font_size=24, color=WHITE)
        num_fuzzy = Text("2,300", font_size=24, color=WHITE)
        group_fuzzy_count = VGroup(label_fuzzy, num_fuzzy).arrange(DOWN, buff=0.1)
        
        # Issue 53 Fix 2: group_crane_count at B3, scale 0.7
        self.place_at_grid(group_crane_count, "B3", scale_factor=0.7)
        # Issue 53 Fix 2: group_fuzzy_count at B5, scale 0.7
        self.place_at_grid(group_fuzzy_count, "B5", scale_factor=0.7)
        
        # Initial candidate clouds (small dots)
        dots_crane = VGroup(*[Dot(radius=0.02, color=COLOR_CRANE, fill_opacity=0.5) for _ in range(50)])
        dots_fuzzy = VGroup(*[Dot(radius=0.02, color=COLOR_FUZZY, fill_opacity=0.5) for _ in range(50)])
        
        for i, dot in enumerate(dots_crane):
            dot.move_to(self.grid["D3"] + np.array([np.random.uniform(-0.6, 0.6), np.random.uniform(-0.6, 0.6), 0]))
        for i, dot in enumerate(dots_fuzzy):
            dot.move_to(self.grid["D5"] + np.array([np.random.uniform(-0.8, 0.8), np.random.uniform(-0.8, 0.8), 0]))

        self.play(
            FadeIn(group_crane_count),
            FadeIn(group_fuzzy_count),
            FadeIn(dots_crane),
            FadeIn(dots_fuzzy)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(COLOR_HIGHLIGHT)
        
        # Countdowns
        # CRANE drops to 60
        # FUZZY drops to 1,500
        
        def update_number(m, alpha, start, end):
            val = int(interpolate(start, end, alpha))
            new_text = Text(f"{val:,}", font_size=24, color=WHITE).match_height(m).move_to(m.get_center())
            m.become(new_text)

        # Shattering effect for CRANE dots
        shatter_anims = []
        for dot in dots_crane:
            target_pos = self.grid["D3"] + np.array([np.random.uniform(-1.5, 1.5), np.random.uniform(-1.5, 1.5), 0])
            shatter_anims.append(dot.animate.move_to(target_pos).set_opacity(0.2))

        self.play(
            UpdateFromAlphaFunc(num_crane, lambda m, a: update_number(m, a, 2300, 60)),
            UpdateFromAlphaFunc(num_fuzzy, lambda m, a: update_number(m, a, 2300, 1500)),
            *shatter_anims,
            run_time=3,
            rate_func=smooth
        )
        
        # Final emphasis
        self.play(
            Indicate(num_crane, color=COLOR_CRANE),
            Flash(num_crane, color=COLOR_CRANE),
            dots_fuzzy.animate.set_color(COLOR_FUZZY).scale(1.2),
            run_time=1.5
        )
        self.wait(2)

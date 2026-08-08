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
        self.setup_layout(
            "The Magic of the Central Limit Theorem: Order from Chaos",
            [
                "Welcome to the mysterious forest of unpredictable monsters.",
                "Monster heights here follow a messy, chaotic distribution.",
                "Some are tiny squirrels, others are giant redwood trees.",
                "Meet Zog, a scientist trying to find the average.",
                "Can we find order in this wild, random data?"
            ]
        )

        # Colors for reference
        MONSTER_COLORS = ["#FF5733", "#33FF57", "#3357FF", "#F333FF", "#FFFF33", 
                          "#33FFFF", "#FF3357", "#57FF33", "#5733FF", "#FF8C00"]
        CHART_COLOR = "#808080"
        HIGHLIGHT_COLOR = "#FFFF00"
        ZOG_COLOR = "#FFFFFF"
        QUESTION_COLOR = "#00FFFF"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        monsters = VGroup()
        heights = [0.6, 0.4, 0.8, 0.7, 0.2, 0.9, 0.5, 0.6, 1.1, 2.0]
        # Revised positions to respect B021 (Column 2+) and B005 (Rows B-D)
        # B2: tiny, D5: giant
        positions = ['B3', 'B4', 'B5', 'C2', 'B2', 'C3', 'C4', 'D2', 'D3', 'D5']
        
        for h, pos, col in zip(heights, positions, MONSTER_COLORS):
            m = Rectangle(height=h, width=0.3, fill_opacity=0.8, color=col)
            self.place_at_grid(m, pos)
            monsters.add(m)
            
        self.play(FadeIn(monsters))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(GRAY),
            self.lecture[1].animate.set_color(CHART_COLOR)
        )
        
        chart_bars = VGroup()
        # Seed for consistency if needed, but Manim generally handles this
        np.random.seed(42)
        for i in range(12):
            h = np.random.uniform(0.3, 1.2)
            b = Rectangle(height=h, width=0.15, fill_opacity=0.6, color=CHART_COLOR)
            chart_bars.add(b)
        chart_bars.arrange(RIGHT, buff=0.1)
        # Issue 21: Change E1 to E2 to prevent crowding
        self.place_in_area(chart_bars, 'E2', 'F6')
        
        self.play(Create(chart_bars))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(GRAY),
            self.lecture[2].animate.set_color(HIGHLIGHT_COLOR)
        )
        
        # Circle tiny at B2, giant at D5
        circle_tiny = Circle(radius=0.2, color=HIGHLIGHT_COLOR).move_to(self.grid['B2'])
        circle_giant = Circle(radius=0.6, color=HIGHLIGHT_COLOR).move_to(self.grid['D5'])
        
        self.play(Create(circle_tiny), Create(circle_giant))
        self.play(circle_tiny.animate.scale(1.2), circle_giant.animate.scale(1.1), rate_func=there_and_back)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(GRAY),
            self.lecture[3].animate.set_color(ZOG_COLOR)
        )
        
        zog_body = Circle(radius=0.25, color=ZOG_COLOR, fill_opacity=0.2)
        glass_l = Circle(radius=0.06, color=ZOG_COLOR).shift(LEFT*0.08 + UP*0.04)
        glass_r = Circle(radius=0.06, color=ZOG_COLOR).shift(RIGHT*0.08 + UP*0.04)
        bridge = Line(glass_l.get_right(), glass_r.get_left(), color=ZOG_COLOR)
        zog_face = VGroup(zog_body, glass_l, glass_r, bridge)
        zog_label = Text("Zog", font_size=18, color=ZOG_COLOR).next_to(zog_body, DOWN, buff=0.1)
        zog_group = VGroup(zog_face, zog_label)
        # Issue 23: Move Zog from C6 to D6
        self.place_at_grid(zog_group, 'D6')
        
        self.play(FadeIn(zog_group))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color(GRAY),
            self.lecture[4].animate.set_color(QUESTION_COLOR)
        )
        
        q_mark = MathTex("?", color=QUESTION_COLOR).scale(2.5)
        # Issue 22: Move question mark from C3-C4 to B4-D5
        self.place_in_area(q_mark, 'B4', 'D5')
        
        # Pulse animation setup using ValueTracker
        pulse_tracker = ValueTracker(1.0)
        # Use add_updater to change scale. Note: scale is relative, so we use a base mobject
        q_mark_base_scale = q_mark.get_height()
        q_mark.add_updater(lambda m: m.scale_to_fit_height(q_mark_base_scale * pulse_tracker.get_value()))

        self.play(
            ReplacementTransform(chart_bars, q_mark),
            FadeOut(monsters),
            FadeOut(circle_tiny),
            FadeOut(circle_giant)
        )
        
        # Pulse effect
        self.play(pulse_tracker.animate.set_value(1.2), run_time=0.5, rate_func=there_and_back)
        self.play(pulse_tracker.animate.set_value(1.2), run_time=0.5, rate_func=there_and_back)
        self.wait(2)

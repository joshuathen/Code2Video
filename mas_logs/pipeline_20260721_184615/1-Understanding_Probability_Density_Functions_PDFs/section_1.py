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
        # Setup the layout with section-specific title and lecture lines
        self.setup_layout("The Hook: Measuring the Infinite", [
            "Meet Pixel, a robot measuring battery life precisely.",
            "Counting discrete objects is simple and exact.",
            "Measuring time involves infinite possible values.",
            "The chance of hitting exactly five hours is zero.",
            "Instead, we look at the probability of ranges."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Meet Pixel, a robot measuring battery life precisely.
        self.play(self.lecture[0].animate.set_color("#32CD32"))
        
        # Assets integration (Issue 19)
        robot = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg", color="#C0C0C0")
        battery_container = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/battery.svg", color="#32CD32")
        
        # Battery fill - using a simple rectangle to animate "draining"
        bat_fill = Rectangle(height=0.4, width=0.8, color="#32CD32", fill_opacity=1, stroke_width=0)
        # Assuming battery icon has a fill area inside.
        battery = VGroup(bat_fill, battery_container)
        
        self.place_at_grid(robot, "B2", scale_factor=0.6)
        self.place_at_grid(battery, "B4", scale_factor=0.8)
        bat_fill.move_to(battery_container.get_left(), aligned_edge=LEFT).shift(RIGHT*0.1)
        
        # ValueTracker for the battery level (Issue 19 - draining battery)
        battery_level = ValueTracker(0.8)
        bat_fill.add_updater(lambda m: m.stretch_to_fit_width(battery_level.get_value(), about_edge=LEFT) if battery_level.get_value() > 0.01 else m.scale(0))
        
        self.play(FadeIn(robot), FadeIn(battery))
        self.play(battery_level.animate.set_value(0.1), run_time=3)
        self.wait(1.5)
        
        # === Animation for Lecture Line 2 ===
        # Counting discrete objects is simple and exact.
        self.play(
            self.lecture[0].animate.set_color("#FFFFFF"),
            self.lecture[1].animate.set_color("#00FFFF")
        )
        
        marbles = VGroup()
        colors = ["#FFD700", "#ADFF2F", "#00FFFF", "#FF69B4", "#FFA500"]
        for i, color in enumerate(colors):
            m = Circle(radius=0.25, color=color, fill_opacity=1)
            # Fallback to Text if MathTex is complex, though numbers are fine.
            label = Text(str(i+1), font_size=18, color="#000000").move_to(m.get_center())
            marbles.add(VGroup(m, label))
        
        marbles.arrange(RIGHT, buff=0.3)
        self.place_in_area(marbles, "D2", "D6", scale_factor=1.0)
        
        self.play(FadeIn(marbles, shift=UP))
        self.wait(2)
        
        # === Animation for Lecture Line 3 ===
        # Measuring time involves infinite possible values.
        self.play(
            self.lecture[1].animate.set_color("#FFFFFF"),
            self.lecture[2].animate.set_color("#FFFFFF"),
            FadeOut(marbles),
            FadeOut(robot),
            FadeOut(battery)
        )
        self.wait(0.5)
        
        timeline = NumberLine(
            x_range=[4, 6, 1],
            length=4,
            include_numbers=True,
            font_size=24,
            label_direction=DOWN,
            color="#FFFFFF"
        )
        # Issue 22: Oversized timeline fix
        self.place_in_area(timeline, "C2", "C6", scale_factor=0.7)
        
        needle = Arrow(start=UP*0.5, end=ORIGIN, buff=0, color="#FFFFFF", max_tip_length_to_length_ratio=0.15)
        needle.next_to(timeline.n2p(5), UP, buff=0.1)
        
        self.play(Create(timeline))
        self.play(GrowArrow(needle))
        self.wait(1.5)
        
        # Zoom effect simulation - preserving relative positioning
        target_timeline = timeline.copy().scale(2).move_to(self.grid["D4"])
        target_needle = Arrow(start=UP*1.0, end=ORIGIN, buff=0, color="#FFFFFF", max_tip_length_to_length_ratio=0.2)
        target_needle.next_to(target_timeline.n2p(5), UP, buff=0.1)
        
        self.play(
            Transform(timeline, target_timeline),
            Transform(needle, target_needle)
        )
        self.wait(2)
        
        # === Animation for Lecture Line 4 ===
        # The chance of hitting exactly five hours is zero.
        self.play(
            self.lecture[2].animate.set_color("#FFFFFF"),
            self.lecture[3].animate.set_color("#FF0000")
        )
        
        # Using Text fallback (L022) or keeping MathTex for simple strings
        prob_text = MathTex("P(X = 5) = 0", color="#FF0000", font_size=42)
        # Issue 23: Reposition to avoid overlap
        self.place_at_grid(prob_text, "B4", scale_factor=0.8)
        
        self.play(
            needle.animate.scale(0),
            FadeIn(prob_text)
        )
        self.wait(2)
        
        # === Animation for Lecture Line 5 ===
        # Instead, we look at the probability of ranges.
        self.play(
            self.lecture[3].animate.set_color("#FFFFFF"),
            self.lecture[4].animate.set_color("#87CEFA")
        )
        
        # Highlight interval [4.5, 5.5] on the scaled timeline
        range_line = Line(
            timeline.n2p(4.5),
            timeline.n2p(5.5),
            color="#87CEFA",
            stroke_width=10
        )
        # Subtle glow effect
        glow = range_line.copy().set_stroke(width=20, opacity=0.25)
        
        range_label = MathTex("P(4.5 \\leq X \\leq 5.5)", color="#87CEFA", font_size=36)
        # Issue 24: Scale range_label to 0.8
        self.place_at_grid(range_label, "B4", scale_factor=0.8)
        
        self.play(
            Create(range_line),
            Create(glow),
            FadeOut(prob_text)
        )
        self.play(Write(range_label))
        self.wait(3.5)

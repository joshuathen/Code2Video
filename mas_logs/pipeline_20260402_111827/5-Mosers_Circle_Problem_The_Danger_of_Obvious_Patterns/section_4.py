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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout(
            "The Great Collapse (n = 6)", 
            [
                "Now, let’s add the sixth point and connect it.", 
                "We expect thirty-two regions, following our doubling rule.", 
                "Let's count them carefully one by one.", 
                "Wait, there are only thirty-one regions here.", 
                "The doubling pattern has completely failed us!"
            ]
        )

        # Colors
        CHORD_COLOR = "#58C4DD"
        HIGHLIGHT_COLOR = "#FFD700"
        FAILURE_COLOR = "#FF0000"
        PREDICTION_COLOR = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Draw circle and points
        circle = Circle(radius=2.0, color=WHITE)
        self.place_in_area(circle, "B2", "E5", scale_factor=1.0)
        
        # Define 6 points with slight offsets to avoid concurrency at center (General Position)
        angles = [10, 75, 130, 195, 255, 310]
        points = [circle.point_at_angle(a * DEGREES) for a in angles]
        dots = VGroup(*[Dot(p, color=BLUE, radius=0.08) for p in points])
        
        chords = VGroup()
        for i in range(6):
            for j in range(i + 1, 6):
                chords.add(Line(points[i], points[j], stroke_width=1.5, color=CHORD_COLOR))
        
        self.play(Create(circle))
        self.play(Create(dots))
        # Draw chords one by one
        for chord in chords:
            self.play(Create(chord), run_time=0.15)
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        prediction_text = Text("32?", font_size=36, color=PREDICTION_COLOR)
        self.place_at_grid(prediction_text, "A4")
        self.play(Write(prediction_text))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Magnifying glass zoom on center - Fix Issue 40: Move to F6
        zoom_circle = Circle(radius=0.9, color=WHITE, stroke_width=2)
        self.place_at_grid(zoom_circle, "F6", scale_factor=1.2)
        zoom_circle.set_fill(BLACK, opacity=1)
        
        # Represent center tiny triangle (visual representation of non-concurrency)
        center_triangle = Triangle(color=HIGHLIGHT_COLOR, fill_opacity=0.6).scale(0.15)
        center_triangle.move_to(zoom_circle.get_center())
        
        zoom_label = Text("Center Zoom", font_size=18, color=WHITE)
        zoom_label.next_to(zoom_circle, UP, buff=0.2)
        
        self.play(FadeIn(zoom_circle), FadeIn(zoom_label))
        self.play(Create(center_triangle))
        self.wait(1)

        # Counter for regions - Using Text as mob_class to avoid LaTeX dependency
        counter_val = Integer(0, color=WHITE, mob_class=Text).scale(1.2)
        self.place_at_grid(counter_val, "B6")
        counter_label = Text("Region Count:", font_size=20, color=WHITE)
        counter_label.next_to(counter_val, UP, buff=0.2)
        
        self.add(counter_label, counter_val)
        
        # Prepare region representative points for highlighting (Manual scattering for n=6)
        c = circle.get_center()
        region_reps = []
        # Outer ring (6 regions)
        for i in range(6):
            angle = (angles[i] + angles[(i+1)%6])/2
            if i == 5: angle += 180 # fix wrapping
            region_reps.append(c + 1.7 * np.array([np.cos(angle*DEGREES), np.sin(angle*DEGREES), 0]))
        # Middle rings (approx 24 points)
        for r in [1.2, 0.7, 0.3]:
            for a in range(0, 360, 45):
                region_reps.append(c + r * np.array([np.cos(a*DEGREES), np.sin(a*DEGREES), 0]))
        # Total approx 31 regions
        region_reps = region_reps[:31]

        # Fast counting animation
        for i in range(1, 32):
            point_idx = i - 1
            flash_dot = Dot(region_reps[point_idx % len(region_reps)], color=HIGHLIGHT_COLOR, radius=0.15)
            self.add(flash_dot)
            counter_val.set_value(i)
            self.wait(0.08)
            self.remove(flash_dot)
        
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Cross out 32?
        cross = Cross(prediction_text, stroke_color=RED, stroke_width=8)
        
        # Spider Max appears - Fix Issue 32 (Asset) and Issue 41 (Positioning)
        # Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/spider.svg
        max_spider = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/spider.svg")
        max_spider.set_color(WHITE)
        self.place_at_grid(max_spider, "F1", scale_factor=0.8)
        
        confused_bubble = Text("?", font_size=32, color=WHITE)
        bubble_circle = Circle(radius=0.25, color=WHITE).move_to(confused_bubble)
        bubble_group = VGroup(bubble_circle, confused_bubble)
        # Fix Issue 41: Positioning
        self.place_at_grid(bubble_group, "E1", scale_factor=0.8)
        
        self.play(Create(cross))
        self.play(FadeIn(max_spider), FadeIn(bubble_group))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Fix Issue 39: Positioning final_31 and pattern_broken to avoid overlap with circle
        final_31 = Text("31", font_size=80, color=FAILURE_COLOR, weight=BOLD)
        self.place_at_grid(final_31, "A2", scale_factor=0.8)
        
        pattern_broken = Text("Pattern Broken!", font_size=36, color=FAILURE_COLOR)
        self.place_at_grid(pattern_broken, "A5", scale_factor=0.8)
        
        self.play(
            FadeIn(final_31, shift=UP),
            Write(pattern_broken)
        )
        self.play(Indicate(final_31, color=FAILURE_COLOR))
        self.wait(3)

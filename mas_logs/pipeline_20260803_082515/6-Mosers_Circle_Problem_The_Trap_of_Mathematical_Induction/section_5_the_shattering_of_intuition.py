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

class Section5TheShatteringOfIntuitionScene(TeachingScene):
    def construct(self):
        self.setup_layout("The Reveal: Counting n = 6", [
            "Let's count the regions for six points carefully.",
            "We are counting every single small piece.",
            "Wait, the final count is only thirty-one regions!",
            "The doubling pattern has finally failed us here.",
            "Mathematical patterns can be deceptive without proof."
        ])

        # === Animation for Lecture Line 1 ===
        # Draw circle with 6 points and all chords (#FFFFFF).
        self.lecture[0].set_color(YELLOW)
        
        circle = Circle(radius=1.8, color=WHITE)
        self.place_in_area(circle, "B2", "E5")
        
        # Define 6 points on the circle
        angles = [i * 60 * DEGREES for i in range(6)]
        points = [circle.point_at_angle(ang) for ang in angles]
        dots = VGroup(*[Dot(p, color=BLUE, radius=0.08) for p in points])
        
        chords = VGroup()
        for i in range(6):
            for j in range(i + 1, 6):
                chords.add(Line(points[i], points[j], color=WHITE, stroke_width=2))

        self.play(Create(circle), run_time=1)
        self.play(Create(dots), run_time=1)
        self.play(Create(chords), run_time=1.5)
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # Color regions one by one from 1 to 31 (#00FF00).
        # A counter on screen increments with each colored region (#FFFF00).
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        counter_val = Integer(0, color="#FFFF00").scale(1.2)
        self.place_at_grid(counter_val, "A5")
        counter_label = Text("Count:", font_size=20, color="#FFFF00")
        counter_label.next_to(counter_val, LEFT, buff=0.2)
        
        self.play(FadeIn(counter_label), FadeIn(counter_val))
        
        # Highlight dots (simulating 31 regions)
        # We distribute 31 points in a way that looks like we are covering the circle's regions.
        region_highlights = VGroup()
        for i in range(1, 32):
            # Spiral distribution to cover internal areas
            phi = i * (2 * np.pi * 0.618) # Golden angle approx
            r = 1.6 * np.sqrt(i / 31)
            p_pos = circle.get_center() + np.array([r * np.cos(phi), r * np.sin(phi), 0])
            
            highlight = Dot(p_pos, color="#00FF00", radius=0.12).set_opacity(0.8)
            region_highlights.add(highlight)
            
            self.play(
                counter_val.animate.set_value(i),
                FadeIn(highlight, scale=0.5),
                run_time=0.12
            )
        
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Wait, the final count is only thirty-one regions!
        # The final number 31 appears large and red (#FF0000).
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        big_31 = Text("31!", font_size=120, color="#FF0000")
        # Issue 35 fix: move to 'A3', 'B4', scale 0.8
        self.place_in_area(big_31, "A3", "B4", scale_factor=0.8)
        
        self.play(Write(big_31))
        self.play(big_31.animate.scale(1.2), run_time=0.3)
        self.play(big_31.animate.scale(1/1.2), run_time=0.3)
        self.play(Indicate(big_31, color="#FF0000"))
        
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # The doubling pattern has finally failed us here.
        # Max icon changes to a shocked expression (#FFD700).
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Max Icon (Shocked Face)
        max_face = Circle(radius=0.4, color="#FFD700", fill_opacity=0.3)
        eye_l = Dot(radius=0.05, color="#FFD700").move_to(max_face.get_center() + LEFT*0.15 + UP*0.1)
        eye_r = Dot(radius=0.05, color="#FFD700").move_to(max_face.get_center() + RIGHT*0.15 + UP*0.1)
        # O-shaped mouth
        mouth = Circle(radius=0.1, color="#FFD700").move_to(max_face.get_center() + DOWN*0.15)
        max_shocked = VGroup(max_face, eye_l, eye_r, mouth)
        
        # Issue 36 fix: move to 'F6'
        self.place_at_grid(max_shocked, "F6")
        
        expected_txt = Text("Expected 32", font_size=18, color=WHITE)
        expected_txt.next_to(max_shocked, UP)
        
        self.play(FadeIn(max_shocked), Write(expected_txt))
        self.play(Flash(max_shocked, color="#FFD700"))
        
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Mathematical patterns can be deceptive without proof.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        lesson_box = Rectangle(height=1, width=4, color=YELLOW)
        lesson_text = Text("Pattern != Proof", font_size=32, color=YELLOW)
        lesson_group = VGroup(lesson_box, lesson_text)
        # Issue 37 fix: move to 'A1', scale 0.7
        self.place_at_grid(lesson_group, "A1", scale_factor=0.7)
        
        self.play(Create(lesson_box), Write(lesson_text))
        self.play(Indicate(lesson_group))
        
        self.wait(2)

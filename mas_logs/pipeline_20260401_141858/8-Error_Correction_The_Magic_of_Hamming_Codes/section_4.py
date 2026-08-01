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

class Section4Scene(TeachingScene):
    def construct(self):
        # Initial Setup
        lecture_lines = [
            "A Venn diagram visualizes the Hamming (7,4) code logic.",
            "Three circles represent the three parity check zones.",
            "Data bits sit in the intersections of these circles.",
            "Bit seven is unique, sitting inside all three circles.",
            "If bit seven flips, all three circles report errors."
        ]
        self.setup_layout("Visualizing Logic: The 3-Circle Venn Diagram", lecture_lines)
        
        # Colors
        color_a = "#FF6347"  # Tomato
        color_b = "#32CD32"  # LimeGreen
        color_c = "#1E90FF"  # DodgerBlue
        highlight_color = YELLOW

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(highlight_color))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Create Circles with centers specifically chosen for overlap
        circle_a = Circle(radius=1.6, color=color_a, fill_opacity=0.1).set_stroke(width=4)
        circle_b = Circle(radius=1.6, color=color_b, fill_opacity=0.1).set_stroke(width=4)
        circle_c = Circle(radius=1.6, color=color_c, fill_opacity=0.1).set_stroke(width=4)
        
        # Position Centers
        self.place_at_grid(circle_a, "C3")
        self.place_at_grid(circle_b, "E2")
        self.place_at_grid(circle_c, "E4")
        
        # Zone Labels
        label_a = Text("Zone A", font_size=16, color=color_a)
        label_b = Text("Zone B", font_size=16, color=color_b)
        label_c = Text("Zone C", font_size=16, color=color_c)
        self.place_at_grid(label_a, "B2")
        self.place_at_grid(label_b, "F1")
        self.place_at_grid(label_c, "F5")

        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(highlight_color),
            Create(circle_a), Create(circle_b), Create(circle_c),
            Write(label_a), Write(label_b), Write(label_c)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Parity Bits 1, 2, 4 (Outer zones)
        p1 = Text("P1", font_size=20, color=color_a)
        p2 = Text("P2", font_size=20, color=color_b)
        p4 = Text("P4", font_size=20, color=color_c)
        self.place_at_grid(p1, "B3")
        self.place_at_grid(p2, "F2")
        self.place_at_grid(p4, "F4")
        
        # Data Bits 3, 5, 6 (Intersection of two circles)
        d3 = Text("D3", font_size=20, color=WHITE) # A & B
        d5 = Text("D5", font_size=20, color=WHITE) # A & C
        d6 = Text("D6", font_size=20, color=WHITE) # B & C
        self.place_at_grid(d3, "D2")
        self.place_at_grid(d5, "D4")
        self.place_at_grid(d6, "E3")
        
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(highlight_color)
        )
        self.play(FadeIn(p1), FadeIn(p2), FadeIn(p4))
        self.play(FadeIn(d3), FadeIn(d5), FadeIn(d6))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Data Bit 7 (Intersection of all three)
        d7 = Text("D7", font_size=22, weight=BOLD, color=WHITE)
        self.place_at_grid(d7, "D3")
        
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(highlight_color)
        )
        self.play(FadeIn(d7))
        self.play(Indicate(d7, color=highlight_color, scale_factor=1.4))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Emphasize shared monitoring and error response
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(RED)
        )
        
        # Flash bit 7 to simulate a flip/error
        self.play(d7.animate.set_color(RED), Flash(d7, color=RED, flash_radius=0.4))
        
        # Show all circles turning red as they detect the center bit flip
        self.play(
            circle_a.animate.set_stroke(color=RED, width=8),
            circle_b.animate.set_stroke(color=RED, width=8),
            circle_c.animate.set_stroke(color=RED, width=8),
            run_time=1
        )
        
        # Final Pulse
        self.play(
            circle_a.animate.set_stroke(width=4),
            circle_b.animate.set_stroke(width=4),
            circle_c.animate.set_stroke(width=4),
            rate_func=there_and_back,
            run_time=1.5
        )
        
        self.wait(2)

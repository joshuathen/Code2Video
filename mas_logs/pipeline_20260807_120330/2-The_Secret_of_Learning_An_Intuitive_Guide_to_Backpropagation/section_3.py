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
        title_text = "Intuition: The Blame Game"
        lecture_lines = [
            "Backpropagation is like a \"blame game\" for errors.",
            "We look at the final mistake and trace backward.",
            "We identify which knobs contributed most to the error.",
            "Each weight gets a share of the \"blame\" signal.",
            "Only those responsible are adjusted to fix the result."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        SUPPLIER_COLOR = "#87CEEB" # Sky Blue
        CHEF_COLOR = "#98FB98"    # Pale Green
        WAITER_COLOR = "#FFDAB9"  # Peach
        ERROR_COLOR = "#FF4500"   # OrangeRed
        HIGHLIGHT_COLOR = "#FFFF00" # Yellow
        CONN_COLOR = "#666666"    # Gray
        
        # --- Build Network Architecture ---
        # Supplier (Input Layer)
        s1 = Circle(radius=0.3, color=SUPPLIER_COLOR, fill_opacity=0.2)
        s2 = Circle(radius=0.3, color=SUPPLIER_COLOR, fill_opacity=0.2)
        self.place_at_grid(s1, "B2")
        self.place_at_grid(s2, "D2")
        l_supplier = Text("Supplier", font_size=18, color=SUPPLIER_COLOR)
        self.place_at_grid(l_supplier, "A2")
        
        # Chef (Hidden Layer)
        c1 = Circle(radius=0.3, color=CHEF_COLOR, fill_opacity=0.2)
        c2 = Circle(radius=0.3, color=CHEF_COLOR, fill_opacity=0.2)
        self.place_at_grid(c1, "B4")
        self.place_at_grid(c2, "D4")
        l_chef = Text("Chef", font_size=18, color=CHEF_COLOR)
        self.place_at_grid(l_chef, "A4")
        
        # Waiter (Output Layer)
        w1 = Circle(radius=0.3, color=WAITER_COLOR, fill_opacity=0.2)
        self.place_at_grid(w1, "C6")
        l_waiter = Text("Waiter", font_size=18, color=WAITER_COLOR)
        self.place_at_grid(l_waiter, "A6") # Fix for Issue 30
        
        # Connections helper
        def get_conn(m1, m2):
            return Line(m1.get_right(), m2.get_left(), color=CONN_COLOR, stroke_width=2)
            
        conns_s_c = VGroup(
            get_conn(s1, c1), get_conn(s1, c2),
            get_conn(s2, c1), get_conn(s2, c2)
        )
        conns_c_w = VGroup(
            get_conn(c1, w1), get_conn(c2, w1)
        )
        
        network = VGroup(s1, s2, c1, c2, w1, conns_s_c, conns_c_w, l_supplier, l_chef, l_waiter)
        self.add(network)
        
        # === Animation for Lecture Line 1 ===
        # Backpropagation is like a "blame game" for errors.
        self.play(self.lecture[0].animate.set_color(HIGHLIGHT_COLOR))
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        # We look at the final mistake and trace backward.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(HIGHLIGHT_COLOR)
        )
        
        # Error indicators at the Waiter (Output)
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/dog.svg] (Issue 22)
        dog_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/dog.svg").scale(0.3)
        error_label = Text("70% Dog", font_size=14, color=ERROR_COLOR)
        error_group = VGroup(dog_icon, error_label).arrange(DOWN, buff=0.1)
        self.place_at_grid(error_group, "E6") # Fix for Issue 32
        
        blame_tag = Text("BLAME", font_size=22, color=ERROR_COLOR, weight=BOLD)
        blame_tag.next_to(w1, RIGHT, buff=0.15)
        
        self.play(
            FadeIn(error_group, shift=UP*0.2),
            Write(blame_tag),
            w1.animate.set_stroke(ERROR_COLOR, width=8).set_fill(ERROR_COLOR, opacity=0.4)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # We identify which knobs contributed most to the error.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(HIGHLIGHT_COLOR)
        )
        
        # Pulses moving backward from Waiter to Chef
        pulse_c1 = Circle(radius=0.1, color=ERROR_COLOR, fill_opacity=1).move_to(w1.get_center())
        pulse_c2 = Circle(radius=0.1, color=ERROR_COLOR, fill_opacity=1).move_to(w1.get_center())
        
        self.play(
            pulse_c1.animate.move_to(c1.get_center()),
            pulse_c2.animate.move_to(c2.get_center()),
            conns_c_w.animate.set_color(ERROR_COLOR).set_stroke(width=4),
            run_time=1.5
        )
        self.play(
            FadeOut(pulse_c1), FadeOut(pulse_c2),
            c1.animate.set_stroke(ERROR_COLOR, width=6),
            c2.animate.set_stroke(ERROR_COLOR, width=6)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Each weight gets a share of the "blame" signal.
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(HIGHLIGHT_COLOR)
        )
        
        # Pulses moving backward from Chef to Supplier
        pulse_s1 = Circle(radius=0.1, color=ERROR_COLOR, fill_opacity=1).move_to(c1.get_center())
        pulse_s2 = Circle(radius=0.1, color=ERROR_COLOR, fill_opacity=1).move_to(c2.get_center())
        
        self.play(
            pulse_s1.animate.move_to(s1.get_center()),
            pulse_s2.animate.move_to(s2.get_center()),
            conns_s_c.animate.set_color(ERROR_COLOR).set_stroke(width=3),
            run_time=1.5
        )
        self.play(
            FadeOut(pulse_s1), FadeOut(pulse_s2),
            s1.animate.set_stroke(ERROR_COLOR, width=4),
            s2.animate.set_stroke(ERROR_COLOR, width=4)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Only those responsible are adjusted to fix the result.
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(HIGHLIGHT_COLOR)
        )
        
        # Highlight Chef 1 (c1) as the responsible "knob"
        highlight_rect = SurroundingRectangle(c1, color=HIGHLIGHT_COLOR, buff=0.1)
        adjust_label = Text("ADJUSTING", font_size=18, color=HIGHLIGHT_COLOR, weight=BOLD)
        self.place_at_grid(adjust_label, "A5", scale_factor=0.6) # Fix for Issue 31
        
        self.play(
            Create(highlight_rect),
            Write(adjust_label),
            c1.animate.set_fill(HIGHLIGHT_COLOR, opacity=0.7).set_stroke(HIGHLIGHT_COLOR, width=8)
        )
        # Animate the Chef knob rotating slightly to show adjustment (Storyboard)
        self.play(
            Indicate(c1, color=HIGHLIGHT_COLOR, scale_factor=1.2),
            Rotate(c1, angle=PI/4)
        )
        self.wait(2)

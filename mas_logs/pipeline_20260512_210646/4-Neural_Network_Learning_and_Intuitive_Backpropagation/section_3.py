from manim import *
import numpy as np
import os

# Pre-emptively handle potential race condition for the 'media/texts' directory
try:
    os.makedirs(os.path.join("media", "texts"), exist_ok=True)
except Exception:
    pass

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
        lecture_lines = [
            "If Nero is wrong, which knob is responsible?",
            "Consider a bucket brigade passing water to a tub.",
            "If water is missing, we look back for spills.",
            "We trace blame backward from the final output.",
            "The chain rule connects these layers of blame."
        ]
        self.setup_layout("The Problem: The 'Chain of Blame'", lecture_lines)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        # Zooms in on nodes + highlight knob asset
        node1 = Circle(radius=0.4, color="#ADD8E6", stroke_width=4)
        node2 = Circle(radius=0.4, color="#ADD8E6", stroke_width=4)
        self.place_at_grid(node1, "C2")
        self.place_at_grid(node2, "C4")
        
        # Asset: knob.svg
        knob = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/knob.svg", color="#C0C0C0")
        self.place_in_area(knob, "B2", "D4", scale_factor=0.6)
        
        wrong_mark = Text("?", color=RED, font_size=60)
        self.place_at_grid(wrong_mark, "B5") # Fixed position as per issue 62
        
        self.play(FadeIn(node1), FadeIn(node2), FadeIn(knob), FadeIn(wrong_mark))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(BLUE)
        
        self.play(FadeOut(node1), FadeOut(node2), FadeOut(knob), FadeOut(wrong_mark))

        # Bucket brigade: 5 blue circles (#ADD8E6)
        circles = VGroup(*[Circle(radius=0.25, color="#ADD8E6", fill_opacity=0.3) for _ in range(5)])
        for i, circle in enumerate(circles):
            self.place_at_grid(circle, f"C{i+1}")
            
        # Blue tub (#1E90FF)
        tub_frame = Rectangle(height=0.6, width=0.8, color="#1E90FF", stroke_width=2)
        tub_fill = Rectangle(height=0.5, width=0.7, color="#1E90FF", fill_opacity=0.8, stroke_width=0)
        self.place_at_grid(tub_frame, "C6")
        tub_fill.move_to(tub_frame.get_center())
        
        tub_label = Text("Tub", font_size=16, color=WHITE)
        self.place_at_grid(tub_label, "E6") # Fixed position as per issue 62
        
        self.play(
            AnimationGroup(*[FadeIn(c) for c in circles], lag_ratio=0.15),
            FadeIn(tub_frame),
            FadeIn(tub_fill),
            FadeIn(tub_label)
        )
        
        # Pulses (#FFFFFF) passing toward tub
        pulses = VGroup(*[Dot(radius=0.08, color="#FFFFFF") for _ in range(5)])
        for i, pulse in enumerate(pulses):
            self.place_at_grid(pulse, f"C{i+1}")
        
        self.play(
            *[pulse.animate.move_to(self.grid[f"C{min(i+2, 6)}"]) for i, pulse in enumerate(pulses)],
            run_time=1.5
        )
        self.play(FadeOut(pulses))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(ORANGE)
        
        # Water level drops
        self.play(tub_fill.animate.scale(np.array([1, 0.4, 1]), about_edge=DOWN))
        
        # Red spill icons (#FF4500)
        spill_icon = Text("!", color="#FF4500", font_size=30)
        self.place_at_grid(spill_icon, "D3") # Fixed position as per issue 62
        
        spill_label = Text("Spill!", font_size=16, color="#FF4500")
        self.place_at_grid(spill_label, "E3") # Fixed position as per issue 62
        
        self.play(FadeIn(spill_icon), FadeIn(spill_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(RED)
        
        # Thick red arrow (#FF0000) backward
        back_arrow = Arrow(
            start=self.grid["C6"],
            end=self.grid["C1"],
            color="#FF0000",
            buff=0.3,
            stroke_width=8
        )
        
        self.play(GrowArrow(back_arrow))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(WHITE)
        
        # 'Chain Rule' text (#FFFFFF) and '×' symbols (#FFFFFF)
        chain_rule_text = Text("Chain Rule", font_size=24, color=WHITE)
        self.place_at_grid(chain_rule_text, "A3")
        
        # '×' symbols between circles
        multipliers = VGroup()
        for i in range(1, 5):
            mult = Text("×", font_size=20, color=WHITE)
            # Position between C_i and C_i+1
            start_pos = self.grid[f"C{i}"]
            end_pos = self.grid[f"C{i+1}"]
            mult.move_to((start_pos + end_pos) / 2 + UP * 0.4)
            multipliers.add(mult)
            
        self.play(
            FadeIn(chain_rule_text),
            AnimationGroup(*[FadeIn(m) for m in multipliers], lag_ratio=0.2)
        )
        
        self.wait(2)

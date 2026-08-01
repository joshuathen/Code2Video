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

class Section5Scene(TeachingScene):
    def construct(self):
        # Assets
        SUN_ASSET = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/sun.svg"
        COMPUTER_ASSET = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg"

        title = "Thermodynamic Limits: Why Math Defies Physics"
        lines = [
            "Flipping bits requires a specific amount of energy.",
            "Computing every hash would consume the Sun's energy.",
            "The oceans would boil before we reach 1 percent.",
            "Security is guarded by the laws of thermodynamics.",
            "This makes 256-bit hashes physically impossible to crack."
        ]
        
        self.setup_layout(title, lines)

        # Colors
        COLOR_ENERGY = "#FFD700"
        COLOR_SUN = "#FFFF00"
        COLOR_OCEAN = "#ADD8E6"
        COLOR_FAIL = "#FF0000"
        COLOR_COMP = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(COLOR_ENERGY))
        
        meter_label_text = Text("Sun Energy Meter", font_size=18, color=COLOR_ENERGY)
        sun_icon_meter = SVGMobject(SUN_ASSET, color=COLOR_ENERGY).scale(0.2)
        meter_header = VGroup(sun_icon_meter, meter_label_text).arrange(RIGHT, buff=0.2)
        self.place_at_grid(meter_header, "A3", scale_factor=1.0)
        
        meter_box = Rectangle(width=4.5, height=0.4, color=COLOR_ENERGY)
        self.place_in_area(meter_box, "B1", "B6")
        
        meter_fill = Rectangle(
            width=4.4, 
            height=0.3, 
            fill_color=COLOR_ENERGY, 
            fill_opacity=1, 
            stroke_width=0
        )
        meter_fill.move_to(meter_box.get_left(), aligned_edge=LEFT).shift(RIGHT * 0.05)
        
        self.play(FadeIn(meter_header), Create(meter_box), FadeIn(meter_fill))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(COLOR_SUN))
        
        # Computer Icon
        computer = SVGMobject(COMPUTER_ASSET, color=COLOR_COMP)
        self.place_at_grid(computer, "C2", scale_factor=0.8)
        
        # Sun Icon
        sun = SVGMobject(SUN_ASSET, color=COLOR_SUN)
        self.place_at_grid(sun, "C5", scale_factor=1.0)
        
        data_flow = Arrow(computer.get_right(), sun.get_left(), color=COLOR_COMP, buff=0.2)
        
        self.play(FadeIn(computer), FadeIn(sun))
        self.play(GrowArrow(data_flow))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(COLOR_OCEAN))
        
        # Meter drops to zero and Sun fades
        self.play(
            meter_fill.animate.scale(0.0001, about_edge=LEFT),
            sun.animate.set_color(BLACK).set_opacity(0.2),
            run_time=3
        )
        
        boil_text = Text("OCEANS BOIL", font_size=32, color=COLOR_OCEAN, weight=BOLD)
        self.place_at_grid(boil_text, "D3", scale_factor=1.2)
        
        steam = VGroup(*[
            Arc(radius=0.15, start_angle=0, angle=PI, color=COLOR_OCEAN).shift(RIGHT*i*0.4)
            for i in range(-5, 6)
        ])
        self.place_in_area(steam, "E1", "E6")
        
        self.play(Write(boil_text), FadeIn(steam, shift=UP))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(COLOR_COMP))
        
        laws_label = Text("Laws of Thermodynamics", font_size=20, color=COLOR_COMP)
        self.place_at_grid(laws_label, "F1", scale_factor=1.0)
        
        shield = Circle(radius=1.8, color=COLOR_COMP, stroke_width=2).set_opacity(0.2)
        self.place_in_area(shield, "B1", "F6")
        
        self.play(FadeIn(laws_label), Create(shield))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(COLOR_FAIL))
        
        fail_percent = Text("0.0001% complete", font_size=24, color=COLOR_FAIL)
        self.place_at_grid(fail_percent, "F5", scale_factor=1.0)
        
        for _ in range(3):
            self.play(fail_percent.animate.set_opacity(0), run_time=0.3)
            self.play(fail_percent.animate.set_opacity(1), run_time=0.3)
            
        self.wait(2)

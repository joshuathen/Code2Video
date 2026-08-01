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

class Section6Scene(TeachingScene):
    def construct(self):
        # Setup the layout with specific lecture lines
        lecture_lines = [
            "Light inside glass hits the boundary at steep angles.",
            "At the critical angle, light can no longer escape.",
            "Total internal reflection keeps data trapped inside fiber cables."
        ]
        self.setup_layout("Extreme Bending: Total Internal Reflection", lecture_lines)
        
        # Define common colors
        GLASS_COLOR = "#E0FFFF"
        RAY_COLOR = YELLOW
        CABLE_COLOR = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Glass medium at the bottom (Rows D, E, F)
        glass_block = Rectangle(width=5.5, height=3, fill_color=GLASS_COLOR, fill_opacity=0.3, stroke_width=1, stroke_color=GLASS_COLOR)
        self.place_in_area(glass_block, "D1", "F6")
        
        # Air boundary
        boundary = Line(self.grid["C1"], self.grid["C6"], color=WHITE)
        air_label = Text("Air (Lower Density)", font_size=18, color=WHITE)
        self.place_at_grid(air_label, "A3") # Issue 48 fix
        
        glass_label = Text("Glass (Higher Density)", font_size=18, color=GLASS_COLOR)
        self.place_at_grid(glass_label, "E1") # Issue 50 fix
        
        # Light Source
        source = Dot(self.grid["F2"], color=RAY_COLOR)
        source_label = Text("Light Source", font_size=16, color=RAY_COLOR)
        self.place_at_grid(source_label, "F1")
        
        # Initial Ray (Steep Angle)
        # From F2 to C3 (Incident)
        incident_ray_1 = Line(self.grid["F2"], self.grid["C3"], color=RAY_COLOR).add_tip(tip_length=0.15)
        # Refracted ray (Bending away from normal) - From C3 to B5
        refracted_ray_1 = Line(self.grid["C3"], self.grid["B5"], color=RAY_COLOR).add_tip(tip_length=0.15)
        
        self.play(FadeIn(glass_block), Create(boundary), FadeIn(source), FadeIn(source_label), FadeIn(air_label), FadeIn(glass_label))
        self.play(Create(incident_ray_1), Create(refracted_ray_1))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color(YELLOW))
        
        # Move incident ray to a more horizontal angle (Critical Angle)
        # From F2 to C4
        incident_ray_2 = Line(self.grid["F2"], self.grid["C4"], color=RAY_COLOR).add_tip(tip_length=0.15)
        # Refracted ray (Now along the surface) - From C4 to C6
        refracted_ray_2 = Line(self.grid["C4"], self.grid["C6"], color=RED).add_tip(tip_length=0.15)
        
        critical_label = Text("Critical Angle: 90 deg", font_size=18, color=RED)
        self.place_at_grid(critical_label, "B5") # Issue 48 fix (and shortened text)

        self.play(
            ReplacementTransform(incident_ray_1, incident_ray_2),
            ReplacementTransform(refracted_ray_1, refracted_ray_2),
            FadeIn(critical_label)
        )
        self.wait(1)
        
        # Even steeper: Total Internal Reflection
        # From F2 to C5
        incident_ray_3 = Line(self.grid["F2"], self.grid["C5"], color=RAY_COLOR).add_tip(tip_length=0.15)
        # Reflected back - From C5 to F6
        reflected_ray_3 = Line(self.grid["C5"], self.grid["F6"], color=RAY_COLOR).add_tip(tip_length=0.15)
        tir_label = Text("Total Internal Reflection!", font_size=20, color=YELLOW)
        self.place_at_grid(tir_label, "D6") # Issue 49 fix

        self.play(
            ReplacementTransform(incident_ray_2, incident_ray_3),
            ReplacementTransform(refracted_ray_2, reflected_ray_3),
            FadeOut(critical_label),
            FadeIn(tir_label)
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color(YELLOW))
        
        # Cleanup previous scene
        self.play(
            FadeOut(glass_block), FadeOut(boundary), FadeOut(source), 
            FadeOut(source_label), FadeOut(air_label), FadeOut(glass_label),
            FadeOut(incident_ray_3), FadeOut(reflected_ray_3), FadeOut(tir_label)
        )

        # Fiber Optic Cable
        cable_top = Line(self.grid["C1"], self.grid["C6"], color=CABLE_COLOR)
        cable_bottom = Line(self.grid["D1"], self.grid["D6"], color=CABLE_COLOR)
        cable_group = VGroup(cable_top, cable_bottom)
        
        # Issue 31: Integrate Asset
        fiber_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/fiber.svg")
        self.place_at_grid(fiber_icon, "A5", scale_factor=0.6)
        
        fiber_label = Text("Fiber Optic Cable", font_size=22, color=CABLE_COLOR)
        self.place_at_grid(fiber_label, "B3")
        
        # Data pulse bouncing through
        pulse = Dot(color=YELLOW, radius=0.1)
        pulse.move_to(self.grid["D1"])
        
        # Path: D1 -> C2 -> D3 -> C4 -> D5 -> C6
        path_points = [self.grid["D1"], self.grid["C2"], self.grid["D3"], self.grid["C4"], self.grid["D5"], self.grid["C6"]]
        
        self.play(Create(cable_group), FadeIn(fiber_label), FadeIn(fiber_icon))
        
        # Show bouncing animation
        self.play(FadeIn(pulse))
        for i in range(len(path_points) - 1):
            self.play(pulse.animate.move_to(path_points[i+1]), run_time=0.4, rate_func=linear)
            
        self.wait(2)

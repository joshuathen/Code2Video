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
        self.setup_layout(
            "Visualizing Solutions & Conclusion",
            [
                "Solutions to PDEs are fields satisfying local rules.",
                "These equations predict weather and keep planes aloft.",
                "PDEs map the dynamic changes of our physical world."
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        # Color coding: Green for fields/flow
        self.play(self.lecture[0].animate.set_color("#00FF00"))
        
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/drone.svg]
        # Resolve Issue 29: Use the drone asset
        drone = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/drone.svg")
        drone.set_color(WHITE)
        self.place_in_area(drone, "B2", "D5", scale_factor=1.0)
        drone_center = drone.get_center()
        
        # Create flow lines (green) representing air pressure fields
        flow_lines = VGroup()
        for offset in np.linspace(-1.5, 1.5, 7):
            start = drone_center + LEFT * 2.5 + UP * offset
            end = drone_center + RIGHT * 2.5 + UP * (offset - 0.2) # Slight downward deflection
            
            # Curvature logic: pull lines slightly around the drone body
            mid_offset = 0.4 if offset > 0 else -0.4
            
            path = CubicBezier(
                start,
                start + RIGHT * 1.5 + UP * mid_offset,
                end + LEFT * 1.5 + UP * mid_offset,
                end,
                color="#00FF00",
                stroke_opacity=0.6
            )
            flow_lines.add(path)

        self.play(FadeIn(drone))
        self.play(
            LaggedStart(
                *[Create(line) for line in flow_lines],
                lag_ratio=0.1,
                run_time=2
            )
        )
        
        # Animate "flow" using moving dots to visualize the "field"
        flow_dots = VGroup()
        for line in flow_lines:
            dot = Dot(color="#00FF00", radius=0.04)
            flow_dots.add(dot)
            
        self.play(
            *[MoveAlongPath(dot, line, rate_func=linear, run_time=2.5) for dot, line in zip(flow_dots, flow_lines)],
        )
        self.remove(flow_dots)
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # Color coding: Cyan for applications
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        
        # Keywords: Weather, Aerodynamics, Physics
        kw_weather = Text("Weather", font_size=24, color="#FFFF00")
        kw_aero = Text("Aerodynamics", font_size=24, color="#00FFFF")
        kw_physics = Text("Physics", font_size=24, color="#FF00FF")
        
        # Resolve Issue 44: place kw_weather at A4
        self.place_at_grid(kw_weather, "A4", scale_factor=0.8)
        # Resolve Issue 45: place kw_aero at D1 and kw_physics at D6
        self.place_at_grid(kw_aero, "D1", scale_factor=0.8)
        self.place_at_grid(kw_physics, "D6", scale_factor=0.8)
        
        self.play(Write(kw_weather))
        self.play(Write(kw_aero))
        self.play(Write(kw_physics))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Final summary line
        self.play(self.lecture[2].animate.set_color(WHITE))
        
        summary_text = Text(
            "PDEs: The Language of the Physical World",
            font_size=28, # Adjusted font size for fit
            color=WHITE,
            weight=BOLD
        )
        
        # Resolve Issue 43: reposition summary_text to E1-F6 to avoid obstruction
        self.place_in_area(summary_text, "E1", "F6", scale_factor=0.8)
        
        self.play(Write(summary_text))
        self.play(Indicate(summary_text, color=YELLOW))
        
        self.wait(3)

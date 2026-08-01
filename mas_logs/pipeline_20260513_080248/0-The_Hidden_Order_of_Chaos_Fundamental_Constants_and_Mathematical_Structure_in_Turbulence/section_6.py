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
        # Setup title and lines
        lines = [
            'These mathematical structures allow us to simulate complex weather.', 
            'Understanding the cascade helps engineers design efficient aircraft wings.', 
            "We find predictable order within the heart of nature's chaos."
        ]
        self.setup_layout("Summary: Predicting the Unpredictable", lines)

        # Colors for matching
        COLOR_LINE1 = "#ADD8E6"  # Light Blue
        COLOR_LINE2 = "#00FFFF"  # Cyan
        COLOR_LINE3 = "#FFFFFF"  # White

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(COLOR_LINE1))

        # Simulation Car (Left side of grid)
        # Using a wireframe representation
        car_points = [
            [-1, -0.3, 0], [1, -0.3, 0], [1, 0, 0], [0.5, 0.4, 0], 
            [-0.5, 0.4, 0], [-1, 0, 0], [-1, -0.3, 0]
        ]
        sim_car = Polygon(*car_points, stroke_width=2, color=WHITE)
        sim_grid = VGroup(*[
            Line(start=[-1.2, y, 0], end=[1.2, y, 0], stroke_width=1, stroke_opacity=0.3)
            for y in np.linspace(-0.5, 0.5, 5)
        ] + [
            Line(start=[x, -0.5, 0], end=[x, 0.5, 0], stroke_width=1, stroke_opacity=0.3)
            for x in np.linspace(-1.2, 1.2, 10)
        ])
        simulation_group = VGroup(sim_grid, sim_car)
        # ISSUE 38: Move and rescale to avoid lecture notes
        self.place_in_area(simulation_group, "B2", "E3", scale_factor=0.6)
        
        sim_label = Text("Simulation", font_size=18, color=COLOR_LINE1)
        # ISSUE 40: Scale by 1.2
        self.place_at_grid(sim_label, "A2", scale_factor=1.2)

        # Physical Car (Right side of grid)
        # ISSUE 28: Use Image Asset
        reality_group = ImageMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/car.png")
        # ISSUE 39: Move and rescale to avoid crowding
        self.place_in_area(reality_group, "B5", "E6", scale_factor=0.6)

        real_label = Text("Reality", font_size=18, color=COLOR_LINE1)
        # ISSUE 40: Scale by 1.2
        self.place_at_grid(real_label, "A5", scale_factor=1.2)

        divider = Line(self.grid["A3"] + RIGHT*0.45, self.grid["F3"] + RIGHT*0.45, color=GRAY, stroke_width=1)

        self.play(
            FadeIn(simulation_group),
            FadeIn(reality_group),
            FadeIn(sim_label),
            FadeIn(real_label),
            Create(divider)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_LINE2)
        )

        # Create cyan streamlines that flow across both panels
        streamline_paths = VGroup()
        for i in range(5):
            y_offset = (i - 2) * 0.3
            # Spans columns 1.5 to 6.5 relative to grid spacing
            start_pt = self.grid["C1"] + LEFT * 0.5 + UP * y_offset
            mid_pt = self.grid["C3"] + UP * (y_offset + 0.2)
            end_pt = self.grid["C6"] + RIGHT * 0.5 + UP * y_offset
            
            path = CubicBezier(start_pt, start_pt + RIGHT*2, end_pt + LEFT*2, end_pt)
            path.set_stroke(COLOR_LINE2, width=2, opacity=0.6)
            streamline_paths.add(path)

        self.play(Create(streamline_paths))
        
        # Animating the flow
        flow_animations = [
            ShowPassingFlash(path.copy().set_stroke(width=4), time_width=0.5, run_time=2, rate_func=linear)
            for path in streamline_paths
        ]
        self.play(*flow_animations)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_LINE3)
        )

        # Fade background/visuals to black (fade out everything except title/lecture)
        self.play(
            FadeOut(simulation_group),
            FadeOut(reality_group),
            FadeOut(sim_label),
            FadeOut(real_label),
            FadeOut(divider),
            FadeOut(streamline_paths)
        )

        # Center final text within the right area
        final_text = Text("Predicting the Unpredictable", font_size=32, color=COLOR_LINE3)
        self.place_in_area(final_text, "A1", "F6")
        self.play(Write(final_text))
        self.wait(2)

        # Final cleanup for smooth ending
        self.play(FadeOut(final_text), FadeOut(self.lecture), FadeOut(self.title))
        self.wait(1)

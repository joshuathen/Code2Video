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
        # Setup title and lecture lines
        lecture_lines = [
            "Refraction splits white light into a vibrant spectrum.",
            "Raindrops act as prisms, creating nature's rainbows.",
            "Light's hidden symphony is revealed through dispersion."
        ]
        self.setup_layout("The Masterpiece: Dispersion in Nature", lecture_lines)

        # Helper for lecture highlighting
        def highlight_line(index):
            self.play(*[self.lecture[i].animate.set_color(WHITE if i != index else YELLOW) for i in range(len(self.lecture))], run_time=0.5)

        # Colors for spectrum
        spectrum_colors = ["#FF0000", "#FF7F00", "#FFFF00", "#00FF00", "#0000FF", "#4B0082", "#8B00FF"]

        # === Animation for Lecture Line 1 ===
        highlight_line(0)
        
        # Prism
        prism = Polygon(
            [-1.5, -1.2, 0], [1.5, -1.2, 0], [0, 1.4, 0],
            color=WHITE, stroke_width=2
        )
        self.place_in_area(prism, "B2", "E4", scale_factor=0.8)
        
        # White ray
        in_point = prism.get_left() + UP * 0.2
        entry_ray = Line(self.grid["B1"] + LEFT * 2, in_point, color=WHITE)
        
        # Spectrum rays (approximated paths)
        exit_point_base = prism.get_right() + DOWN * 0.5
        spectrum_rays = VGroup()
        for i, color in enumerate(spectrum_colors):
            # Spread slightly more for violet than red
            offset = i * 0.15
            target = self.grid["E6"] + RIGHT * 2 + DOWN * offset
            ray = Line(in_point, exit_point_base + UP * (0.2 - offset*0.5), color=color)
            out_ray = Line(exit_point_base + UP * (0.2 - offset*0.5), target, color=color)
            spectrum_rays.add(VGroup(ray, out_ray))

        self.play(Create(prism))
        self.play(Create(entry_ray))
        self.play(AnimationGroup(*[Create(sr) for sr in spectrum_rays], lag_ratio=0.1))
        self.wait(1)
        
        self.play(FadeOut(prism), FadeOut(entry_ray), FadeOut(spectrum_rays))

        # === Animation for Lecture Line 2 ===
        highlight_line(1)
        
        # Raindrop
        raindrop = Circle(radius=1.5, color="#ADD8E6", stroke_width=4)
        self.place_in_area(raindrop, "B2", "E5", scale_factor=0.8)
        
        # Ray-trace through raindrop (Simplified)
        # Entry (Upper left)
        entry_start = self.grid["A1"] + LEFT
        entry_hit = raindrop.point_at_angle(150 * DEGREES)
        white_in = Line(entry_start, entry_hit, color=WHITE)
        
        # Inside (Refraction + Dispersion to back)
        internal_rays = VGroup()
        exit_rays = VGroup()
        for i, color in enumerate(spectrum_colors):
            angle_spread = i * 0.02
            back_hit = raindrop.point_at_angle((0 - angle_spread) * DEGREES)
            exit_hit = raindrop.point_at_angle((240 + angle_spread*5) * DEGREES)
            
            ray_to_back = Line(entry_hit, back_hit, color=color)
            ray_to_exit = Line(back_hit, exit_hit, color=color)
            ray_out = Line(exit_hit, exit_hit + (DOWN + RIGHT*0.5) * 2, color=color)
            
            internal_rays.add(VGroup(ray_to_back, ray_to_exit))
            exit_rays.add(ray_out)

        self.play(Create(raindrop))
        self.play(Create(white_in))
        self.play(Create(internal_rays))
        self.play(Create(exit_rays))
        self.wait(1)

        self.play(FadeOut(raindrop), FadeOut(white_in), FadeOut(internal_rays), FadeOut(exit_rays))

        # === Animation for Lecture Line 3 ===
        highlight_line(2)
        
        # Rainbow arc
        rainbow_arc = VGroup()
        for i, color in enumerate(spectrum_colors):
            arc = Arc(
                radius=2.0 + i*0.15,
                start_angle=0,
                angle=PI,
                color=color,
                stroke_width=10
            )
            rainbow_arc.add(arc)
        
        self.place_in_area(rainbow_arc, "B1", "F6", scale_factor=0.8)
        rainbow_arc.rotate(PI) # Flip to make it look like a hill

        # Final Title
        final_title = Text("The Hidden Symphony of Light", font_size=36, color=WHITE)
        self.place_in_area(final_title, "C2", "C5", scale_factor=1.0)
        final_title.set_z_index(10)

        self.play(Create(rainbow_arc), run_time=2)
        self.play(Write(final_title))
        self.wait(3)

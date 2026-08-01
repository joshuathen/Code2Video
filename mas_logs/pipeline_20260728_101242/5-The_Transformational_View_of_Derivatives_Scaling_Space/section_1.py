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

class Section1Scene(TeachingScene):
    def construct(self):
        # Setup data from storyboard
        title = "Introduction: Beyond the Slope"
        lecture_lines = [
            "Think of functions as maps between number lines.",
            "Input space connects to output space with elastic strings.",
            "Derivatives represent the dynamic movement of these strings."
        ]
        self.setup_layout(title, lecture_lines)

        # Asset path
        string_asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/string.svg"

        # === Animation for Lecture Line 1 ===
        # "Think of functions as maps between number lines."
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Create Input and Output number lines
        # Using B2-B6 for Input and E2-E6 for Output as per standard parallel lines view
        input_line = Line(self.grid["B2"], self.grid["B6"], color=WHITE)
        output_line = Line(self.grid["E2"], self.grid["E6"], color=WHITE)
        
        # Labels for the spaces
        input_label = Text("Input x", font_size=20, color="#BBBBBB")
        output_label = Text("Output f(x)", font_size=20, color="#BBBBBB")
        
        # Resolve Issue 22: use place_in_area for input_label at B1
        self.place_in_area(input_label, "B1", "B1", scale_factor=0.7)
        # Resolve Issue 23: use place_in_area for output_label at E1
        self.place_in_area(output_label, "E1", "E1", scale_factor=0.6)
        
        self.play(Create(input_line), Create(output_line))
        self.play(Write(input_label), Write(output_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Input space connects to output space with elastic strings."
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        # Define source dots on Input line
        src_keys = ["B2", "B3", "B4", "B5"]
        # Define target dots on Output line (showing scaling/transformation)
        tgt_keys = ["E2", "E4", "E5", "E6"]
        
        dots_top = VGroup(*[Dot(self.grid[k], color=BLUE_B) for k in src_keys])
        dots_bottom = VGroup(*[Dot(self.grid[k], color=BLUE_B) for k in tgt_keys])
        
        # Helper to align SVG to points without using put_start_and_end_on (which can fail on VGroups)
        def safe_align_svg(svg, start, end):
            # Check if SVG has points to avoid empty object errors
            has_points = any(len(sub.points) > 0 for sub in svg.family_members_with_points())
            if not has_points:
                # Fallback to a simple line if asset loading fails or is empty
                return Line(start, end)
            
            # Reset position
            svg.move_to(ORIGIN)
            # Calculate transformation
            vec = end - start
            angle = np.arctan2(vec[1], vec[0])
            dist = np.linalg.norm(vec)
            
            # Apply transformations
            # Assume default asset orientation is horizontal.
            if svg.width > 0:
                svg.scale(dist / svg.width)
            svg.rotate(angle)
            svg.move_to((start + end) / 2)
            return svg

        # Resolve Issue 19: Use SVGMobject [Asset: .../string.svg] for "elastic strings"
        strings = VGroup()
        for s_key, t_key in zip(src_keys, tgt_keys):
            s_svg_raw = SVGMobject(string_asset_path)
            s_svg = safe_align_svg(s_svg_raw, self.grid[s_key], self.grid[t_key])
            s_svg.set_color(WHITE).set_stroke(opacity=0.5)
            strings.add(s_svg)
        
        self.play(FadeIn(dots_top))
        self.play(Create(strings), FadeIn(dots_bottom))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Derivatives represent the dynamic movement of these strings."
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # Highlight one mapping: f(x)
        # Highlight B3 -> E4
        hl_index = 1
        start_p = self.grid[src_keys[hl_index]]
        end_p = self.grid[tgt_keys[hl_index]]
        
        hl_circle_top = Circle(radius=0.15, color="#FFFF00").move_to(start_p)
        hl_circle_bottom = Circle(radius=0.15, color="#FFFF00").move_to(end_p)
        
        # Resolve Issue 19: Use SVGMobject [Asset: .../string.svg] for the highlighted string
        hl_svg_raw = SVGMobject(string_asset_path)
        hl_string = safe_align_svg(hl_svg_raw, start_p, end_p)
        hl_string.set_color("#FFFF00")
        # Ensure it's visually distinct
        if isinstance(hl_string, VMobject):
            hl_string.set_stroke(width=4)
        
        mapping_label = Text("f(x)", font_size=20, color="#FFFF00")
        # Resolve Issue 24: Move mapping_label from C4 to C5 to avoid clutter
        self.place_at_grid(mapping_label, "C5", scale_factor=0.7)

        self.play(
            Create(hl_circle_top),
            Create(hl_circle_bottom),
            Create(hl_string),
            Write(mapping_label)
        )
        self.wait(2)
        
        # Cleanup colors
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)

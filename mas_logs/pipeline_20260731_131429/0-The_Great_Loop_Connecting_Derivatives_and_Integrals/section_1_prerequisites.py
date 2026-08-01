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

class Section1PrerequisitesScene(TeachingScene):
    def construct(self):
        # Data from storyboard
        title = "Prerequisite Check: The Slope and the Area"
        lecture_lines = [
            "Calculus studies slope and area applied to curves.",
            "Slope measures the steepness at a single point.",
            "Area measures the space accumulated under the curve."
        ]
        
        self.setup_layout(title, lecture_lines)
        
        # Colors
        COLOR_HILL = "#00FF00"
        COLOR_SNAIL = "#FFFFFF"
        COLOR_SLOPE = "#FFD700"
        COLOR_AREA = "#90EE90"

        # === Animation for Lecture Line 1 ===
        # Line 1: Calculus studies slope and area applied to curves.
        self.play(self.lecture[0].animate.set_color(COLOR_HILL))
        
        # Hill curve: local parabola y = -0.5*(x-2)^2 + 1
        hill_curve = ParametricFunction(
            lambda t: np.array([t, -0.5 * (t - 2)**2 + 1, 0]),
            t_range=[0, 4],
            color=COLOR_HILL
        )
        
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/snail.svg]
        snail = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/snail.svg")
        snail.set_color(COLOR_SNAIL)
        snail.scale(0.3)
        snail.move_to(hill_curve.point_from_proportion(0.25))
        
        # Group and position in the right-side area
        hill_vgroup = VGroup(hill_curve, snail)
        # Resolved Issue 21: Change hill_vgroup scale factor to 0.9
        self.place_in_area(hill_vgroup, "B1", "E6", scale_factor=0.9)
        
        snail_label = Text("Snail", font_size=18, color=COLOR_SNAIL)
        # Resolved Issue 22: Move snail_label to A2
        self.place_at_grid(snail_label, "A2", scale_factor=0.8)
        
        self.play(Create(hill_curve))
        self.play(FadeIn(snail), Write(snail_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Line 2: Slope measures the steepness at a single point.
        self.play(self.lecture[1].animate.set_color(COLOR_SLOPE))
        
        # Tangent at snail's position (using proportion 0.25 on the original curve)
        tangent_line = TangentLine(hill_curve, alpha=0.25, length=2, color=COLOR_SLOPE)
        
        slope_label = Text("Slope", font_size=24, color=COLOR_SLOPE)
        # Resolved Issue 23: Move slope_label to B2
        self.place_at_grid(slope_label, "B2", scale_factor=0.8)
        
        self.play(Create(tangent_line))
        self.play(Write(slope_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Line 3: Area measures the space accumulated under the curve.
        self.play(self.lecture[2].animate.set_color(COLOR_AREA))
        
        # Create area fill under the curve
        # Get points from the curve (already scaled/positioned by hill_vgroup)
        curve_points = hill_curve.get_points()
        bottom_y = curve_points[0][1] - 0.5 
        
        # Construct the polygon for the area
        poly_points = [
            *curve_points,
            [curve_points[-1][0], bottom_y, 0],
            [curve_points[0][0], bottom_y, 0]
        ]
        
        area_fill = Polygon(
            *poly_points, 
            color=COLOR_AREA, 
            fill_opacity=0.3, 
            stroke_width=0
        )
        
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/grass.svg]
        grass_icons = VGroup()
        for i in range(3):
            grass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grass.svg")
            grass.set_color(COLOR_AREA)
            grass.scale(0.15)
            # Spread grass across the area
            pos = area_fill.get_center() + np.array([(i-1)*0.8, -0.3, 0])
            grass.move_to(pos)
            grass_icons.add(grass)

        area_label = Text("Area", font_size=24, color=COLOR_AREA)
        # Place label inside the shaded area at D4
        self.place_at_grid(area_label, "D4", scale_factor=0.8)
        
        self.play(FadeIn(area_fill), FadeIn(grass_icons))
        self.play(Write(area_label))
        self.wait(2)

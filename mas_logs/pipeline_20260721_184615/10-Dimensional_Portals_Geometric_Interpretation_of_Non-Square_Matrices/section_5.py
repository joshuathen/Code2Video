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
        # Data from storyboard
        title = "The Null Space: The Lost Dimension"
        lines = [
            "Some 3D directions collapse to zero during projection.",
            "The null space contains all vectors that disappear.",
            "A whole line or plane might vanish entirely.",
            "It is the hidden dimension of the transformation.",
            "Where movement is swallowed by the origin."
        ]
        
        # Initialize layout
        self.setup_layout(title, lines)
        
        # Visual anchor: the origin for our 3D-to-2D projection
        origin_pos = self.grid["C4"]
        
        # Asset: Load the Null Space icon
        null_icon_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/null.svg"
        null_icon = SVGMobject(null_icon_path).set_color("#FFFF00")
        
        # Projection helper (isometric style for 2D representation of 3D)
        def project_3d(point):
            x, y, z = point
            # Isometric-like projection: Z axis at 45 degrees up-right
            return np.array([x + 0.5 * z, y + 0.5 * z, 0])

        # === Animation for Lecture Line 1 ===
        # L1: Some 3D directions collapse to zero during projection.
        # A1: In 3D, draw a line #FFFF00 representing the Null Space [Asset: ...] passing through the origin.
        self.lecture[0].set_color("#FFFF00")
        
        # Create 3D axes
        x_axis = Line(project_3d([-2, 0, 0]), project_3d([2, 0, 0]), color="#666666", stroke_width=2)
        y_axis = Line(project_3d([0, -2, 0]), project_3d([0, 2, 0]), color="#666666", stroke_width=2)
        z_axis = Line(project_3d([0, 0, -2]), project_3d([0, 0, 2]), color="#666666", stroke_width=2)
        
        axes_label_x = Text("X", font_size=14, color="#666666").move_to(origin_pos + project_3d([2.2, 0, 0]))
        axes_label_y = Text("Y", font_size=14, color="#666666").move_to(origin_pos + project_3d([0, 2.2, 0]))
        axes_label_z = Text("Z", font_size=14, color="#666666").move_to(origin_pos + project_3d([0, 0, 2.2]))

        axes_group = VGroup(x_axis, y_axis, z_axis)
        axes_group.shift(origin_pos)
        
        # Null space line along the Z axis
        null_line = Line(project_3d([0, 0, -1.5]), project_3d([0, 0, 1.5]), color="#FFFF00", stroke_width=6)
        null_line.shift(origin_pos)
        
        # Place Null icon at the top of the line initially
        self.place_at_grid(null_icon, "B5", scale_factor=0.3)
        
        self.play(Create(axes_group), FadeIn(axes_label_x), FadeIn(axes_label_y), FadeIn(axes_label_z), run_time=1.5)
        self.play(Create(null_line), FadeIn(null_icon), run_time=1.2)
        self.wait(2.0)

        # === Animation for Lecture Line 2 ===
        # L2: The null space contains all vectors that disappear.
        # A2: Apply the 2x3 transformation: the entire line #FFFF00 collapses into a single dot at the origin.
        self.lecture[1].set_color("#FFFF00")
        
        # Dot at the center representing the collapsed state
        origin_dot = Dot(origin_pos, radius=0.08, color="#FFFF00")
        
        # Simulate collapse: Z dimension goes to 0
        self.play(
            null_line.animate.scale(0).move_to(origin_pos),
            null_icon.animate.scale(0).move_to(origin_pos),
            z_axis.animate.set_stroke(opacity=0.1),
            axes_label_z.animate.set_fill(opacity=0.1),
            FadeIn(origin_dot),
            run_time=2.0
        )
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        # L3: A whole line or plane might vanish entirely.
        # A3: Show several vectors #00FFFF on that line all moving to (0,0) simultaneously.
        self.lecture[2].set_color("#00FFFF")
        
        # Briefly show "ghost" vectors on the null line before they collapse
        ghost_line = Line(project_3d([0, 0, -1.5]), project_3d([0, 0, 1.5]), color="#FFFF00", stroke_opacity=0.2)
        ghost_line.shift(origin_pos)
        
        vector_z_values = [-1.4, -0.7, 0.7, 1.4]
        vectors = VGroup(*[
            Arrow(origin_pos, origin_pos + project_3d([0, 0, zv]), buff=0, color="#00FFFF", stroke_width=4)
            for zv in vector_z_values
        ])
        
        self.play(FadeIn(ghost_line), Create(vectors))
        self.wait(1.5)
        
        # All these vectors "disappear" into the origin
        self.play(
            *[v.animate.scale(0, about_point=origin_pos) for v in vectors],
            run_time=2.0
        )
        self.remove(ghost_line)
        self.wait(2.0)

        # === Animation for Lecture Line 4 ===
        # L4: It is the hidden dimension of the transformation.
        # A4: Label the origin 'Zero Vector' #FFFFFF and the line 'Null Space' [Asset: ...] #FFFF00.
        self.lecture[3].set_color("#FFFFFF")
        
        # Labels according to Proximity Rule (L002) and VideoCritic fixes
        zero_label = Text("Zero Vector", font_size=18, color="#FFFFFF")
        # Issue 34 Fix
        self.place_in_area(zero_label, "E3", "E5", scale_factor=0.8)
        
        null_label = Text("Null Space", font_size=18, color="#FFFF00")
        # Issue 33 Fix
        self.place_in_area(null_label, "B1", "B3", scale_factor=0.8)
        
        # Null icon next to label
        null_icon_2 = SVGMobject(null_icon_path).set_color("#FFFF00").scale(0.3)
        null_icon_2.next_to(null_label, LEFT, buff=0.1)
        
        # Indicator arrow pointing from the label to the origin
        indicator = Arrow(self.grid["B3"] + DOWN*0.2, origin_pos, color="#FFFF00", buff=0.1, stroke_width=3)
        
        self.play(
            Write(zero_label),
            Write(null_label),
            FadeIn(null_icon_2),
            Create(indicator),
            run_time=1.5
        )
        self.wait(2.0)

        # === Animation for Lecture Line 5 ===
        # L5: Where movement is swallowed by the origin.
        # A5: Highlight the lost Z-dimension #0000FF as it disappears during the transform.
        self.lecture[4].set_color("#0000FF")
        
        # Visual highlight of the axis that vanished
        z_highlight = Line(project_3d([0, 0, -2]), project_3d([0, 0, 2]), color="#0000FF", stroke_width=8)
        z_highlight.shift(origin_pos).set_stroke(opacity=0.6)
        
        self.play(FadeIn(z_highlight))
        # Use 'Indicate' (L004)
        self.play(Indicate(z_highlight, color="#0000FF"))
        self.wait(1.0)
        
        # Final visual of the lost dimension being "swallowed"
        self.play(
            z_highlight.animate.scale(0).move_to(origin_pos),
            run_time=2.0
        )
        
        self.wait(3.0)

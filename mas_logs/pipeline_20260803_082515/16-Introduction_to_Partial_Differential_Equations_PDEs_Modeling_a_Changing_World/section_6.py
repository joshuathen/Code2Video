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
        # Setup layout with title and lecture lines
        self.setup_layout(
            "Real-World Impact: Beyond the Textbook",
            [
                "PDEs drive modern weather prediction models.",
                "They simulate everything from flight to finance.",
                "These equations are tools for global modeling."
            ]
        )
        
        # Define colors for lecture lines
        line_colors = ["#66C2FF", "#99FF99", "#FF99FF"]

        # === Animation for Lecture Line 1 ===
        # Show a grid over a map [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/map.svg] with moving wind vectors.
        self.lecture[0].set_color(line_colors[0])
        
        # Load map asset
        try:
            map_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/map.svg")
        except:
            # Fallback if asset is missing
            map_svg = Rectangle(width=3, height=2)
            
        map_svg.set_color(GRAY).set_opacity(0.4)
        
        # Create a background area and grid lines
        map_bg = Rectangle(width=4, height=3, fill_color="#1A3A5A", fill_opacity=0.2, stroke_color=WHITE, stroke_width=0.5)
        map_svg.scale_to_fit_width(3.5).move_to(map_bg.get_center())
        
        grid_lines = VGroup()
        for i in range(5):
            grid_lines.add(Line(map_bg.get_left() + RIGHT * i, map_bg.get_left() + RIGHT * i + UP * 1.5 + DOWN * 1.5, stroke_width=0.3, color=GRAY))
        for i in range(4):
            grid_lines.add(Line(map_bg.get_bottom() + UP * i, map_bg.get_bottom() + UP * i + RIGHT * 2 + LEFT * 2, stroke_width=0.3, color=GRAY))
        
        map_container = VGroup(map_bg, map_svg, grid_lines)
        # Fix per Issue 32: Relocate to avoid crowding
        self.place_in_area(map_container, 'B4', 'E6', scale_factor=0.8)
        
        # Create wind vectors over the map
        vectors = VGroup()
        rows_v, cols_v = 3, 4
        # Calculate bounds from map_bg after placement
        ul = map_bg.get_corner(UL)
        dr = map_bg.get_corner(DR)
        w = dr[0] - ul[0]
        h = ul[1] - dr[1]
        
        for r in range(rows_v):
            for c in range(cols_v):
                start_pos = ul + RIGHT * (c + 0.5) * (w / cols_v) + DOWN * (r + 0.5) * (h / rows_v)
                vec = Vector(RIGHT * 0.3, color=line_colors[0], stroke_width=2)
                vec.move_to(start_pos)
                vectors.add(vec)
        
        # Animate vectors moving/rotating slightly to represent wind flow
        def update_vectors(vgroup, dt):
            t = self.renderer.time
            for i, vec in enumerate(vgroup):
                noise = np.sin(t + i * 0.7) * 0.3
                vec.set_angle(noise)
                # Subtle length change
                vec.scale(1.0 + 0.05 * np.cos(t + i), about_point=vec.get_start())

        vectors.add_updater(update_vectors)
        
        self.play(FadeIn(map_container), FadeIn(vectors), run_time=2)
        self.wait(3)
        
        # Transition out
        self.play(FadeOut(map_container), FadeOut(vectors), run_time=1)
        self.lecture[0].set_color(WHITE)

        # === Animation for Lecture Line 2 ===
        # Display a fluctuating line graph representing financial markets.
        self.lecture[1].set_color(line_colors[1])
        
        axes = Axes(
            x_range=[0, 10, 2],
            y_range=[0, 5, 1],
            x_length=4,
            y_length=2.5,
            axis_config={"color": WHITE, "include_tip": False},
        ).scale(0.8)
        
        graph_label_x = Text("Time", font_size=14).next_to(axes.x_axis, DOWN, buff=0.1)
        graph_label_y = Text("Value", font_size=14).rotate(PI/2).next_to(axes.y_axis, LEFT, buff=0.1)
        graph_group = VGroup(axes, graph_label_x, graph_label_y)
        
        # Fix per Issue 33: Relocate for better balance
        self.place_in_area(graph_group, 'B4', 'E6')
        
        # Fluctuating path
        time_tracker = ValueTracker(0)
        market_line = VMobject(color=line_colors[1])
        
        def update_market_line(mobj):
            t_max = time_tracker.get_value()
            if t_max <= 0:
                mobj.set_points([])
                return
            points = []
            for t in np.linspace(0, t_max, max(2, int(t_max * 15))):
                # Simulated fluctuation: trend + noise
                val = 1.5 + 0.15 * t + 0.4 * np.sin(t * 1.5) + 0.2 * np.cos(t * 4)
                points.append(axes.c2p(t, val))
            if len(points) >= 2:
                mobj.set_points_as_corners(points)
        
        market_line.add_updater(update_market_line)
        
        self.play(Create(graph_group), run_time=1)
        self.add(market_line)
        self.play(time_tracker.animate.set_value(10), run_time=5, rate_func=linear)
        self.wait(2)
        
        # Transition out
        self.play(FadeOut(graph_group), FadeOut(market_line), run_time=1)
        self.lecture[1].set_color(WHITE)

        # === Animation for Lecture Line 3 ===
        # A complex, evolving simulation mesh fills the entire screen area.
        self.lecture[2].set_color(line_colors[2])
        
        # Create a grid of dots and connect them
        mesh_dots = VGroup()
        grid_size_x, grid_size_y = 6, 6
        
        # Initial points in a local coordinate system
        points_matrix = []
        for i in range(grid_size_y):
            row_points = []
            for j in range(grid_size_x):
                p = np.array([j - (grid_size_x-1)/2, (grid_size_y-1)/2 - i, 0]) * 0.6
                dot = Dot(p, radius=0.03, color=line_colors[2], fill_opacity=0.6)
                dot.original_pos = p.copy()
                mesh_dots.add(dot)
                row_points.append(dot)
            points_matrix.append(row_points)
            
        mesh_lines = VGroup()
        for i in range(grid_size_y):
            for j in range(grid_size_x):
                if j < grid_size_x - 1:
                    mesh_lines.add(Line(points_matrix[i][j].get_center(), points_matrix[i][j+1].get_center(), stroke_width=1, color=line_colors[2], stroke_opacity=0.4))
                if i < grid_size_y - 1:
                    mesh_lines.add(Line(points_matrix[i][j].get_center(), points_matrix[i+1][j].get_center(), stroke_width=1, color=line_colors[2], stroke_opacity=0.4))
        
        full_mesh = VGroup(mesh_dots, mesh_lines)
        # Fix per Issue 34: Relocate to avoid encroaching on lecture notes
        self.place_in_area(full_mesh, 'A4', 'F6', scale_factor=0.9)
        
        # Update function to evolve the mesh
        def update_mesh(mgroup, dt):
            t = self.renderer.time
            dots = mgroup[0]
            lines = mgroup[1]
            
            # Update dot positions relative to their initial centers (after placement)
            # We need to account for the fact that dot.original_pos is in local coords
            # but dot.get_center() is in world coords. 
            # A simpler way: just offset current world pos slightly.
            for dot in dots:
                # Use a wave-like pattern
                pos = dot.get_center()
                shift = np.array([
                    0.01 * np.sin(t + pos[0]),
                    0.01 * np.cos(t + pos[1]),
                    0
                ])
                dot.shift(shift)
            
            # Update line connections
            line_idx = 0
            for i in range(grid_size_y):
                for j in range(grid_size_x):
                    if j < grid_size_x - 1:
                        lines[line_idx].put_start_and_end_on(points_matrix[i][j].get_center(), points_matrix[i][j+1].get_center())
                        line_idx += 1
                    if i < grid_size_y - 1:
                        lines[line_idx].put_start_and_end_on(points_matrix[i][j].get_center(), points_matrix[i+1][j].get_center())
                        line_idx += 1

        full_mesh.add_updater(update_mesh)
        
        self.play(FadeIn(full_mesh), run_time=2)
        self.wait(5)
        
        # Final transition
        full_mesh.remove_updater(update_mesh)
        self.play(FadeOut(full_mesh), run_time=2)
        self.lecture[2].set_color(WHITE)
        self.wait(2)

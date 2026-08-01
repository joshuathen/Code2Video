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

class Section4Scene(TeachingScene):
    def construct(self):
        # Metadata
        title_text = "The Integral: Slicing and Stacking"
        lecture_lines = [
            "Finding the area of wavy, irregular shapes is difficult.",
            "We slice the area into many thin vertical strips.",
            "Each strip is treated as a simple rectangle.",
            "We sum the areas of these infinitely thin rectangles.",
            "This total sum gives us the exact area under curves."
        ]
        
        self.setup_layout(title_text, lecture_lines)
        
        # Define Colors
        WAVY_COLOR = "#00FF00"
        RECT_COLOR = "#00FFFF"
        LABEL_COLOR = "#FFFF00"
        TEXT_COLOR = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WAVY_COLOR)
        
        # Asset: Load the wall SVG
        wall_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/wall.svg")
        wall_svg.set_color(WAVY_COLOR)
        wall_svg.set_fill(WAVY_COLOR, opacity=0.3)
        
        # Create Axes for the graph (Positioned on the right side)
        axes = Axes(
            x_range=[0, 4.2, 1],
            y_range=[0, 4.2, 1],
            x_length=4.2,
            y_length=3.2,
            axis_config={"include_tip": False, "include_ticks": False, "color": GREY_D}
        )
        
        # Define a wavy curve function for rectangles logic
        def wavy_func(x):
            return 0.6 * np.sin(1.5 * x) + 0.3 * np.cos(3 * x) + 2.5
            
        graph = axes.plot(wavy_func, x_range=[0, 4], color=WAVY_COLOR)
        
        # Group and position the visualization in the right-side area
        visual_group = VGroup(axes, wall_svg, graph)
        self.place_in_area(visual_group, "B2", "E5")
        
        self.play(FadeIn(wall_svg), Create(graph))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(RECT_COLOR)
        
        # Initial coarse Riemann rectangles
        rects = axes.get_riemann_rectangles(graph, x_range=[0, 4], dx=0.5, color=RECT_COLOR, fill_opacity=0.6)
        
        self.play(FadeIn(rects))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(RECT_COLOR)
        
        # Highlight a specific rectangle
        sample_rect = rects[3]
        self.play(Indicate(sample_rect, color=RECT_COLOR, scale_factor=1.2))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(RECT_COLOR)
        
        # Increase the number of rectangles (Mid-step)
        rects_mid = axes.get_riemann_rectangles(graph, x_range=[0, 4], dx=0.15, color=RECT_COLOR, fill_opacity=0.6)
        self.play(Transform(rects, rects_mid))
        self.wait(0.5)
        
        # Increase even further (Fine-step)
        rects_fine = axes.get_riemann_rectangles(graph, x_range=[0, 4], dx=0.04, color=RECT_COLOR, fill_opacity=0.6)
        self.play(Transform(rects, rects_fine))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(TEXT_COLOR)
        
        # Total Area Label inside the shape - Move to A4 per VideoCritic feedback
        total_area_label = Text("Total Area", font_size=24, color=LABEL_COLOR)
        self.place_at_grid(total_area_label, "A4", scale_factor=0.8)
        
        # "Integration" text below the visualization - Adjusted per VideoCritic feedback
        integration_text = Text("Integration", font_size=32, color=TEXT_COLOR)
        self.place_in_area(integration_text, "F2", "F5", scale_factor=0.8)
        
        self.play(
            FadeIn(total_area_label),
            FadeIn(integration_text),
            # Final look: make wall solid and rects faint
            rects.animate.set_fill(opacity=0.3),
            wall_svg.animate.set_fill(opacity=0.8)
        )
        self.wait(2)

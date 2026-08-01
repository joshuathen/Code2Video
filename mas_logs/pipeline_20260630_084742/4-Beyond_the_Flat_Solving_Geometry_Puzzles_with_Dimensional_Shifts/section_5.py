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

class Section5Scene(TeachingScene):
    def construct(self):
        # Setup title and lines
        lines = [
            'We can imagine the fourth dimension using these same steps.',
            'A tesseract is a cube extended into another dimension.',
            'It appears as eight cubes folded into one another.',
            'Shadows and nets help us map 4D into 3D.',
            'Mastering dimensions lets us solve the impossible.'
        ]
        self.setup_layout("The 4th Dimension Leap (Summary)", lines)

        # Helper to create a 2D projection of a cube
        def create_cube_wireframe(side=1.0, color=WHITE, offset=0.3):
            front = Square(side_length=side, color=color)
            back = Square(side_length=side, color=color).shift(RIGHT*offset + UP*offset)
            connectors = VGroup(*[
                Line(front.get_vertices()[i], back.get_vertices()[i], color=color)
                for i in range(4)
            ])
            return VGroup(back, connectors, front)

        # === Animation for Lecture Line 1 ===
        # Show a 3D net of 8 white cube frames (#FFFFFF) arranged in a cross shape, incorporating net asset.
        self.lecture[0].set_color(WHITE)
        net_cubes = VGroup()
        # Create 8 cubes in a "hypercube net" shape (cross)
        net_positions = [
            (-1.5, 0), (-0.5, 0), (0.5, 0), (1.5, 0), # Row
            (-0.5, 1), (-0.5, -1), # Column
            (-0.5, 2), (0.5, 1) # Extra branches
        ]
        
        for pos in net_positions:
            c = create_cube_wireframe(side=0.4, color=WHITE, offset=0.1)
            c.move_to(RIGHT * pos[0] * 0.7 + UP * pos[1] * 0.7)
            net_cubes.add(c)
        
        # Load and place net asset [Asset: /mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/nets.svg]
        nets_svg = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/nets.svg").set_color(WHITE)
        self.place_at_grid(nets_svg, "A5", scale_factor=0.6)
        
        self.place_in_area(net_cubes, "B2", "E5", scale_factor=0.8)
        self.play(FadeIn(net_cubes), FadeIn(nets_svg))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Animate the cubes merging into a 2D projection of a Tesseract, utilizing the shadows asset.
        self.lecture[1].set_color(WHITE)
        
        # Schlegel diagram (inner cube inside outer cube)
        inner_side = 0.8
        outer_side = 2.0
        inner_square = Square(side_length=inner_side, color=WHITE)
        outer_square = Square(side_length=outer_side, color=WHITE)
        connectors = VGroup(*[
            Line(inner_square.get_vertices()[i], outer_square.get_vertices()[i], color=WHITE)
            for i in range(4)
        ])
        tesseract = VGroup(inner_square, outer_square, connectors)
        # Issue 45: Using scale factor 0.9 for better margins
        self.place_in_area(tesseract, "B2", "E5", scale_factor=0.9)
        
        # Store center for updater logic later
        tesseract_center = tesseract.get_center()
        
        # Load shadows asset [Asset: /mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/shadows.svg]
        shadows_svg = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/shadows.svg").set_color(WHITE)
        self.place_at_grid(shadows_svg, "A5", scale_factor=0.6)
        
        self.play(
            ReplacementTransform(net_cubes, tesseract),
            ReplacementTransform(nets_svg, shadows_svg)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Visualizing the 8 cubes by pulsing the connections/faces
        self.lecture[2].set_color(WHITE)
        self.play(tesseract.animate.set_stroke(width=4), run_time=0.5)
        self.play(tesseract.animate.set_stroke(width=2), run_time=0.5)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Highlight connectors in light blue (#ADD8E6).
        self.lecture[3].set_color("#ADD8E6")
        highlight_color = "#ADD8E6"
        
        self.play(connectors.animate.set_color(highlight_color), run_time=1)
        
        # Issue 44: Fix label overlap by moving to F3-F5 area
        net_label = Text("Net Projection", font_size=18, color=WHITE)
        self.place_in_area(net_label, "F3", "F5", scale_factor=0.7)
        self.play(Write(net_label))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Tesseract 'rotates' by inner cube expanding and outer cube shrinking.
        # Entire structure pulses in light blue.
        self.lecture[4].set_color(highlight_color)
        
        # Value trackers for the rotation effect
        s1 = ValueTracker(inner_side)
        s2 = ValueTracker(outer_side)
        
        def update_tesseract(mob):
            new_inner = Square(side_length=s1.get_value(), color=WHITE)
            new_outer = Square(side_length=s2.get_value(), color=WHITE)
            new_conns = VGroup(*[
                Line(new_inner.get_vertices()[i], new_outer.get_vertices()[i], color=highlight_color)
                for i in range(4)
            ])
            mob[0].become(new_inner)
            mob[1].become(new_outer)
            mob[2].become(new_conns)
            mob.move_to(tesseract_center)

        tesseract.add_updater(update_tesseract)
        
        # Expand inner, shrink outer (4D rotation loop)
        self.play(
            s1.animate.set_value(outer_side),
            s2.animate.set_value(inner_side),
            run_time=2,
            rate_func=linear
        )
        self.play(
            s1.animate.set_value(inner_side),
            s2.animate.set_value(outer_side),
            run_time=2,
            rate_func=linear
        )
        
        tesseract.remove_updater(update_tesseract)
        
        # Final Pulse
        self.play(
            tesseract.animate.scale(1.1).set_color(highlight_color),
            rate_func=there_and_back,
            run_time=1.5
        )
        
        self.wait(2)

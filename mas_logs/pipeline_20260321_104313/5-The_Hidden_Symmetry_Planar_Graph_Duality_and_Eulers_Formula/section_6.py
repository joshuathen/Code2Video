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
        # Initial layout setup
        self.setup_layout("Application: From Map Coloring to Circuit Design", [
            "Duality simplifies problems like the Four Color Theorem.",
            "It helps engineers optimize paths in electrical circuit design.",
            "This hidden symmetry unifies geometry and network logic."
        ])

        # === Animation for Lecture Line 1 ===
        # Highlight lecture line 1
        self.play(self.lecture[0].animate.set_color("#00FFFF"))

        # Create a simple "map" using 4 colored regions
        region1 = Square(side_length=1, fill_opacity=0.6, color=BLUE_E, stroke_width=2)
        region2 = Square(side_length=1, fill_opacity=0.6, color=RED_E, stroke_width=2)
        region3 = Square(side_length=1, fill_opacity=0.6, color=GREEN_E, stroke_width=2)
        region4 = Square(side_length=1, fill_opacity=0.6, color=ORANGE, stroke_width=2)
        
        map_group = VGroup(
            VGroup(region1, region2).arrange(RIGHT, buff=0),
            VGroup(region3, region4).arrange(RIGHT, buff=0)
        ).arrange(DOWN, buff=0)
        
        # Positioned starting at B2 to E5 to avoid overcrowding title
        self.place_in_area(map_group, "B2", "E5", scale_factor=1.0)
        self.play(FadeIn(map_group))

        # Show the dual graph vertices (at the center of each region)
        # Manually align to grid for strict adherence to positioning rules
        v1 = Dot(color="#00FFFF")
        v2 = Dot(color="#00FFFF")
        v3 = Dot(color="#00FFFF")
        v4 = Dot(color="#00FFFF")
        
        # Positioning vertices to align with the regions in B2-E5
        self.place_at_grid(v1, "C3")
        self.place_at_grid(v2, "C4")
        self.place_at_grid(v3, "D3")
        self.place_at_grid(v4, "D4")
        
        # Dual edges connecting adjacent regions
        e1 = Line(v1.get_center(), v2.get_center(), color="#00FFFF")
        e2 = Line(v1.get_center(), v3.get_center(), color="#00FFFF")
        e3 = Line(v2.get_center(), v4.get_center(), color="#00FFFF")
        e4 = Line(v3.get_center(), v4.get_center(), color="#00FFFF")
        
        dual_graph = VGroup(v1, v2, v3, v4, e1, e2, e3, e4)
        
        self.play(FadeIn(dual_graph))
        self.play(map_group.animate.set_opacity(0.15))
        self.wait(2)
        self.play(FadeOut(map_group), FadeOut(dual_graph))

        # === Animation for Lecture Line 2 ===
        # Reset line 1, highlight line 2
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#00FF00")
        )

        # Create a circuit diagram (two loops/meshes)
        circuit_outline = Rectangle(width=3, height=2, color=WHITE)
        circuit_divider = Line(UP, DOWN, color=WHITE).scale(1.0) # Matches rect height
        circuit = VGroup(circuit_outline, circuit_divider)
        self.place_in_area(circuit, "B2", "E5", scale_factor=1.2)

        # Labels for the meshes (loops) - positioned above the nodes to avoid overlap
        mesh_a_label = Text("Mesh A", font_size=16, color=WHITE)
        mesh_b_label = Text("Mesh B", font_size=16, color=WHITE)
        self.place_at_grid(mesh_a_label, "B3")
        self.place_at_grid(mesh_b_label, "B4")

        self.play(Create(circuit))
        self.play(Write(mesh_a_label), Write(mesh_b_label))
        self.wait(1)

        # Show the dual nodes representing the meshes
        node_a = Dot(color="#00FF00")
        node_b = Dot(color="#00FF00")
        self.place_at_grid(node_a, "C3")
        self.place_at_grid(node_b, "C4")
        
        # Dual edge representing the shared component between meshes
        dual_wire = Line(node_a.get_center(), node_b.get_center(), color="#00FF00")
        
        self.play(
            FadeIn(node_a, node_b),
            Create(dual_wire),
            circuit.animate.set_stroke(opacity=0.3),
            mesh_a_label.animate.set_opacity(0.3),
            mesh_b_label.animate.set_opacity(0.3)
        )
        self.wait(2)
        self.play(FadeOut(circuit), FadeOut(node_a), FadeOut(node_b), FadeOut(dual_wire), FadeOut(mesh_a_label), FadeOut(mesh_b_label))

        # === Animation for Lecture Line 3 ===
        # Reset line 2, highlight line 3
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#FFD700")
        )

        # Display closing text in Gold
        closing_text = Text("Symmetry in Structure", color="#FFD700", font_size=36)
        self.place_in_area(closing_text, "B2", "E5")
        
        self.play(Write(closing_text))
        self.wait(3)

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
        # Initialize Layout
        title = "Summary & Real-World Connection"
        lines = [
            "Matrix dimensions dictate the jump between spaces.",
            "From neural networks to computer graphics, this is key.",
            "Non-square matrices are our portals between dimensions."
        ]
        self.setup_layout(title, lines)
        
        # === Animation for Lecture Line 1 ===
        # Display text 'N-dimensions' #00FF00, an arrow #FFFFFF, and 'M-dimensions' #FF00FF.
        self.play(self.lecture[0].animate.set_color("#00FF00"))
        
        text_n = Text("N-dimensions", color="#00FF00")
        text_m = Text("M-dimensions", color="#FF00FF")
        arrow = Arrow(start=LEFT, end=RIGHT, color="#FFFFFF", buff=0.1)
        
        # Positions adjusted per Issue 35
        self.place_at_grid(text_n, 'A1', scale_factor=0.6)
        self.place_at_grid(text_m, 'A6', scale_factor=0.6)
        self.place_in_area(arrow, 'A2', 'A5', scale_factor=0.8)
        
        self.play(Write(text_n), Write(text_m), Create(arrow))
        self.wait(2.0)

        # === Animation for Lecture Line 2 ===
        # Show a schematic of a neural network layer with 3 input nodes and 2 output nodes.
        self.play(
            self.lecture[0].animate.set_color("#FFFFFF"),
            self.lecture[1].animate.set_color("#FF00FF")
        )
        
        # Nodes: 3 input, 2 output
        in_nodes = VGroup(*[Circle(radius=0.15, color="#00FFFF", fill_opacity=0.5, fill_color="#00FFFF") for _ in range(3)])
        out_nodes = VGroup(*[Circle(radius=0.15, color="#FF00FF", fill_opacity=0.5, fill_color="#FF00FF") for _ in range(2)])
        
        # Positions adjusted per Issue 36
        self.place_at_grid(in_nodes[0], 'B1')
        self.place_at_grid(in_nodes[1], 'C1')
        self.place_at_grid(in_nodes[2], 'D1')
        
        self.place_at_grid(out_nodes[0], 'B3')
        self.place_at_grid(out_nodes[1], 'D3')
        
        # Create connectivity lines
        connections = VGroup()
        for i_node in in_nodes:
            for o_node in out_nodes:
                line = Line(i_node.get_center(), o_node.get_center(), color="#FFFFFF", stroke_width=1.5, stroke_opacity=0.5)
                connections.add(line)
        
        self.play(Create(in_nodes), Create(out_nodes))
        self.play(Create(connections))
        self.wait(2.0)

        # === Animation for Lecture Line 3 ===
        # A 3D mesh model [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/mesh.svg] #00FFFF morphs into a 2D pixel grid #FFFFFF on a screen.
        self.play(
            self.lecture[1].animate.set_color("#FFFFFF"),
            self.lecture[2].animate.set_color("#00FFFF")
        )
        
        # Integration per Issue 21: Load the mesh SVG asset
        mesh_3d = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/mesh.svg", color="#00FFFF")
        
        # Positions adjusted per Issue 37
        self.place_in_area(mesh_3d, 'D4', 'F6', scale_factor=1.0)
        
        # Create a 2D Pixel Grid representing the screen
        grid_2d = VGroup()
        for x in np.linspace(-1, 1, 6):
            grid_2d.add(Line(np.array([x, -1, 0]), np.array([x, 1, 0]), color="#FFFFFF", stroke_width=1.5))
        for y in np.linspace(-1, 1, 6):
            grid_2d.add(Line(np.array([-1, y, 0]), np.array([1, y, 0]), color="#FFFFFF", stroke_width=1.5))
        
        # Position grid per Issue 37
        self.place_in_area(grid_2d, 'D4', 'F6', scale_factor=1.0)
        
        self.play(Create(mesh_3d))
        self.wait(1.5)
        
        # Morphing the mesh into the 2D grid
        self.play(ReplacementTransform(mesh_3d, grid_2d))
        self.wait(3.0)

        # Final cleanup
        self.play(
            FadeOut(grid_2d), 
            FadeOut(text_n), FadeOut(text_m), FadeOut(arrow),
            FadeOut(in_nodes), FadeOut(out_nodes), FadeOut(connections),
            self.lecture[2].animate.set_color("#FFFFFF")
        )
        self.wait(2.0)

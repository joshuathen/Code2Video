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
        # Setup the layout with title and lecture lines
        lecture_lines = [
            "These curves have powerful applications in modern computer science.",
            "They map 2D data onto a 1D linear sequence.",
            "Crucially, points near each other in 2D stay nearby.",
            "This locality is vital for efficient database indexing.",
            "It also enables high-performance image compression and storage."
        ]
        self.setup_layout("Real-World Application: Locality-Preserving Hashing", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Display 'Database' and 'Image' labels
        self.lecture[0].set_color(WHITE)
        db_label = Text("Database", font_size=24, color="#FFFFFF")
        img_label = Text("Image", font_size=24, color="#FFFFFF")
        self.place_at_grid(db_label, "A2")
        self.place_at_grid(img_label, "A5")
        
        self.play(Write(db_label), Write(img_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # A 2D grid of dots traversed by a yellow Hilbert curve
        self.lecture[0].set_color(GRAY)
        self.lecture[1].set_color("#FFFF00")
        
        # Hilbert path coordinates for a 4x4 grid (Order 2)
        hilbert_indices = [
            (0,0), (0,1), (1,1), (1,0), 
            (2,0), (3,0), (3,1), (2,1), 
            (2,2), (3,2), (3,3), (2,3), 
            (1,3), (1,2), (0,2), (0,3)
        ]
        
        # Create dots at the relative grid points
        dots = VGroup(*[Dot(radius=0.1, color=WHITE).move_to(np.array([x, y, 0])) for x, y in hilbert_indices])
        
        # Create the Hilbert curve
        curve = VMobject(color="#FFFF00")
        curve.set_points_as_corners([d.get_center() for d in dots])
        
        # Combine into a group and place in the designated visual area
        viz_group = VGroup(dots, curve)
        self.place_in_area(viz_group, "B2", "E5", scale_factor=0.8)
        
        self.play(Create(dots))
        self.play(Create(curve), run_time=3)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Two neighboring points in the 2D grid are highlighted
        self.lecture[1].set_color(GRAY)
        self.lecture[2].set_color("#FF00FF")
        
        # We pick indices 5 and 6 in the sequence: (3,0) and (3,1)
        high_dot1 = Dot(dots[5].get_center(), radius=0.15, color="#FF00FF")
        high_dot2 = Dot(dots[6].get_center(), radius=0.15, color="#FF00FF")
        
        self.play(Flash(high_dot1, color="#FF00FF"), Flash(high_dot2, color="#FF00FF"))
        self.add(high_dot1, high_dot2)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Show 'Index Blocks' grouping nearby points
        self.lecture[2].set_color(GRAY)
        self.lecture[3].set_color("#00FF00")
        
        idx_label = Text("Index Blocks", font_size=24, color="#00FF00")
        self.place_at_grid(idx_label, "F3")
        
        # Create visual blocks (rectangles) around segments of the curve
        rects = VGroup()
        for i in range(0, 16, 4):
            group = VGroup(dots[i], dots[i+1], dots[i+2], dots[i+3])
            rect = SurroundingRectangle(group, color="#00FF00", buff=0.1, stroke_width=2)
            rects.add(rect)
            
        self.play(Write(idx_label), Create(rects))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # The 2D grid becomes a pixel array (image compression/storage)
        self.lecture[3].set_color(GRAY)
        self.lecture[4].set_color("#00FFFF")
        
        # Calculate pixel size based on the current dot spacing
        side = np.linalg.norm(dots[1].get_center() - dots[0].get_center())
        
        pixels = VGroup()
        for i in range(16):
            pixel_color = interpolate_color(BLUE, GREEN, i / 15)
            sq = Square(side_length=side, fill_opacity=0.7, fill_color=pixel_color, stroke_width=1, color=WHITE)
            sq.move_to(dots[i].get_center())
            pixels.add(sq)
        
        # Ensure the curve is on top to show the traversal order
        curve.set_z_index(10)
        
        self.play(
            FadeOut(rects),
            FadeOut(high_dot1),
            FadeOut(high_dot2),
            FadeOut(idx_label),
            ReplacementTransform(dots, pixels)
        )
        self.wait(2)

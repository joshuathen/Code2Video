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

class Section4Scene(TeachingScene):
    def construct(self):
        # Fetch title and lecture lines from storyboard
        title = "Integration: The Magic of 'Slicing Up'"
        lecture_lines = [
            "Integration helps us find the area of curvy shapes.",
            "Imagine slicing a wavy shape into many thin rectangles.",
            "Each thin rectangle has a simple area to calculate.",
            "Adding all these tiny areas gives the total area.",
            "Thinner slices lead to a perfectly accurate total."
        ]
        
        self.setup_layout(title, lecture_lines)
        
        # Colors
        leaf_color = "#32CD32" # Lime Green
        rect_color = "#FFFF00" # Yellow
        label_color = "#FFFFFF" # White
        highlight_color = "#FFD700" # Gold

        # Helper for leaf shape
        def get_leaf_shape():
            res = 60
            # Parametric-ish leaf shape
            t = np.linspace(0, 1, res)
            # Upper curve: starts at (-1.5, 0), ends at (1.5, 0)
            x_upper = -1.5 + 3 * t
            y_upper = 1.2 * np.sin(np.pi * t) + 0.3 * np.sin(2 * np.pi * t)
            # Lower curve
            x_lower = 1.5 - 3 * t
            y_lower = - (0.8 * np.sin(np.pi * t) + 0.2 * np.sin(2 * np.pi * t))
            
            points = []
            for i in range(res):
                points.append(np.array([x_upper[i], y_upper[i], 0]))
            for i in range(res):
                points.append(np.array([x_lower[i], y_lower[i], 0]))
            
            leaf = VMobject()
            leaf.set_points_as_corners(points)
            leaf.set_color(leaf_color)
            leaf.set_stroke(width=4)
            return leaf

        leaf_outline = get_leaf_shape()
        # Fix Issue 34 & 35: Positioning to avoid occlusion and utilize vertical space
        self.place_in_area(leaf_outline, 'B3', 'F6', scale_factor=0.85)
        
        # Helper for rectangles
        def get_rectangles(n, color=rect_color, opacity=0.4, stroke_width=1):
            rects = VGroup()
            center = leaf_outline.get_center()
            width = leaf_outline.width
            dx = width / n
            
            # Find bounds of the leaf to sample height accurately
            left_x = leaf_outline.get_left()[0]
            
            for i in range(n):
                x_mid = left_x + (i + 0.5) * dx
                
                # To find the height at x_mid, we find the intersection of the vertical line at x_mid with the leaf
                # For simplicity, we use the same curve logic as get_leaf_shape
                t = (x_mid - (center[0] - 1.5 * 0.85)) / (3 * 0.85) # Inverse mapping approx
                # Actually, easier to just use the points on the leaf_outline
                # But even easier: define the functions separately
                
                def get_h(x_val):
                    # Local x relative to leaf center, scaled by 0.85
                    lx = (x_val - center[0]) / 0.85
                    if lx < -1.5 or lx > 1.5: return 0
                    tn = (lx + 1.5) / 3
                    yu = 1.2 * np.sin(np.pi * tn) + 0.3 * np.sin(2 * np.pi * tn)
                    yl = - (0.8 * np.sin(np.pi * tn) + 0.2 * np.sin(2 * np.pi * tn))
                    return (yu - yl) * 0.85

                h = get_h(x_mid)
                if h < 0.05: h = 0.05 # Minimum height for visibility
                
                rect = Rectangle(
                    width=dx, 
                    height=h, 
                    fill_color=color, 
                    fill_opacity=opacity, 
                    stroke_width=stroke_width, 
                    stroke_color=WHITE
                )
                rect.move_to(np.array([x_mid, center[1] + (get_h(x_mid)/2 + (-0.8 * np.sin(np.pi * ((x_mid - (center[0] - 1.5 * 0.85)) / (3 * 0.85))) - 0.2 * np.sin(2 * np.pi * ((x_mid - (center[0] - 1.5 * 0.85)) / (3 * 0.85)))) * 0.85), 0]))
                # Correct vertical alignment is tricky. Let's align bottom to the lower curve
                tn_val = (x_mid - (center[0] - 1.5 * 0.85)) / (3 * 0.85)
                y_low_val = center[1] - (0.8 * np.sin(np.pi * tn_val) + 0.2 * np.sin(2 * np.pi * tn_val)) * 0.85
                rect.align_to(np.array([x_mid, y_low_val, 0]), DOWN)
                rects.add(rect)
            return rects

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(leaf_color)
        self.play(Create(leaf_outline), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE) # Dim previous
        self.lecture[1].set_color(rect_color)
        rects_5 = get_rectangles(5, opacity=0.4, stroke_width=2)
        self.play(Create(rects_5), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE) # Dim previous
        self.lecture[2].set_color(label_color)
        # Choose a representative rectangle (middle one)
        target_rect = rects_5[2]
        
        label_w = MathTex("W", font_size=24, color=label_color)
        label_h = MathTex("H", font_size=24, color=label_color)
        
        label_w.next_to(target_rect, DOWN, buff=0.1)
        label_h.next_to(target_rect, RIGHT, buff=0.1)
        
        self.play(
            target_rect.animate.set_fill(opacity=0.7).set_stroke(color=label_color, width=3),
            Write(label_w),
            Write(label_h),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE) # Dim previous
        self.lecture[3].set_color(rect_color)
        # Highlight rectangles sequentially to show summation
        self.play(FadeOut(label_w), FadeOut(label_h))
        self.play(
            Succession(
                *[rect.animate.set_fill(opacity=0.8, color=highlight_color).set_fill(opacity=0.4, color=rect_color) for rect in rects_5]
            ),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE) # Dim previous
        self.lecture[4].set_color(rect_color)
        rects_100 = get_rectangles(100, opacity=0.6, stroke_width=0) # Set stroke_width to 0 for large N
        self.play(ReplacementTransform(rects_5, rects_100), run_time=3)
        self.wait(2)
